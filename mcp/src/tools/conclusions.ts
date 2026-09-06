import { z } from "zod";

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, workspaceIdSchema } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── list_conclusions ────────────────────────────────────────────────
  server.registerTool(
    "list_conclusions",
    {
      description: [
        "List conclusions (facts and observations) that Honcho has derived about a peer (paginated).",
        "Use this to see what Honcho has learned. If no target is given, returns self-conclusions.",
        "Returns conclusion objects with pagination metadata.",
        "Pass include_deleted to also see retired conclusions and why each was removed.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: list conclusions about this target. Omit for self-conclusions.",
          ),
        include_deleted: z
          .boolean()
          .optional()
          .describe(
            "Include retired conclusions, each carrying the reason it was removed.",
          ),
        page: z
          .number()
          .int()
          .min(1)
          .optional()
          .describe("Page number, 1-indexed. Retired conclusions past page one are only reachable this way."),
        size: z
          .number()
          .int()
          .min(1)
          .max(100)
          .optional()
          .describe("Results per page (default 50, max 100)."),
        filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Additional filter criteria, e.g. {"level": "explicit"}. Merged with this peer pair.',
          ),
      },
    },
    async ({
      workspace_id,
      peer_id,
      target_peer_id,
      include_deleted,
      page: pageNumber,
      size,
      filters,
    }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = target_peer_id
          ? peer.conclusionsOf(target_peer_id)
          : peer.conclusions;
        const page = await scope.list({
          includeDeleted: include_deleted,
          page: pageNumber,
          size,
          filters,
        });
        return textResult({
          conclusions: page.items.map((c) => ({
            id: c.id,
            content: c.content,
            observer_id: c.observerId,
            observed_id: c.observedId,
            session_id: c.sessionId,
            times_derived: c.timesDerived,
            created_at: c.createdAt,
            ...(c.deletedAt
              ? { deleted_at: c.deletedAt, removal: c.removal ?? null }
              : {}),
          })),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to list conclusions: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── query_conclusions ───────────────────────────────────────────────
  server.registerTool(
    "query_conclusions",
    {
      description: [
        "Semantic search across a peer's conclusions.",
        "Use this to find specific knowledge Honcho has derived — more targeted than list_conclusions.",
        "Returns an array of matching conclusions ranked by relevance.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        query: z.string().describe("Semantic search query."),
        target_peer_id: z
          .string()
          .optional()
          .describe("Optional: search conclusions about this target."),
        top_k: z
          .number()
          .optional()
          .describe("Max results to return."),
        filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Optional: filter criteria, e.g. {"level": ["deductive", "inductive"]} to only return conclusions derived during dreaming. Levels: explicit (extracted directly from messages), deductive, inductive, contradiction. See https://honcho.dev/docs/v3/documentation/features/advanced/using-filters',
          ),
      },
    },
    async ({ workspace_id, peer_id, query, target_peer_id, top_k, filters }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = target_peer_id
          ? peer.conclusionsOf(target_peer_id)
          : peer.conclusions;
        const conclusions = await scope.query(query, top_k, undefined, filters);
        return textResult(
          conclusions.map((c) => ({
            id: c.id,
            content: c.content,
            level: c.level,
            observer_id: c.observerId,
            observed_id: c.observedId,
            session_id: c.sessionId,
            times_derived: c.timesDerived,
            created_at: c.createdAt,
          })),
        );
      } catch (e) {
        return errorResult(
          `Query failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── create_conclusions ──────────────────────────────────────────────
  server.registerTool(
    "create_conclusions",
    {
      description: [
        "Manually create conclusions (facts/observations) about a peer.",
        "Use this to inject knowledge into Honcho that wasn't derived from conversation.",
        "Returns the number of conclusions created.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .describe("The peer the conclusions are about."),
        conclusions: z.array(
          z.object({
            content: z.string(),
            action: z.enum(["create", "enrich"]),
            target_id: z.string().optional(),
            reason_for_entry: z.string(),
            search_query: z.string(),
            searched_conclusion_ids: z.array(z.string()),
            source_message_ids: z.array(z.number().int()).optional(),
            times_derived: z
              .number()
              .int()
              .min(1)
              .optional()
              .describe(
                "Rarely needed. OMIT when consolidating: retiring each source as duplicate_absorbed already moves its derivation count onto the survivor, so supplying a count here counts those sources twice. Enrichment carries the predecessor's count forward on its own.",
              ),
          }),
        ).describe("Search-backed admission decisions. Search with query_conclusions before writing."),
        agent_trace_id: z.string().describe("Agent run or trace identifier."),
        agent_model: z.string().describe("Exact model identifier making the decision."),
        session_id: z
          .string()
          .optional()
          .describe(
            "Optional: associate conclusions with a session. Omit for global conclusions.",
          ),
      },
    },
    async ({
      workspace_id,
      peer_id,
      target_peer_id,
      conclusions,
      agent_trace_id,
      agent_model,
      session_id,
    }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = peer.conclusionsOf(target_peer_id);
        const sourceToolCallId = `mcp-create-conclusion-${crypto.randomUUID()}`;
        const params = conclusions.map((item) => ({
          content: item.content,
          sessionId: session_id,
          action: item.action,
          targetId: item.target_id,
          reasonForEntry: item.reason_for_entry,
          searchQuery: item.search_query,
          searchedConclusionIds: item.searched_conclusion_ids,
          sourceMessageIds: item.source_message_ids,
          sourceToolCallId: sourceToolCallId,
          entryOrigin: "explicit_agent" as const,
          agentTraceId: agent_trace_id,
          agentModel: agent_model,
          timesDerived: item.times_derived,
        }));
        const created = await scope.create(params);
        return textResult({
          created: created.length,
          conclusions: created.map((c) => ({
            id: c.id,
            content: c.content,
            times_derived: c.timesDerived,
          })),
        });
      } catch (e) {
        return errorResult(
          `Failed to create conclusions: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── delete_conclusion ───────────────────────────────────────────────
  server.registerTool(
    "delete_conclusion",
    {
      description: [
        "Retire a specific conclusion by ID, recording why.",
        "Use query_conclusions or list_conclusions to find the ID first.",
        "The row is kept with your reason; it stops being returned by search and stops",
        "reaching any deriving agent. When another conclusion carries the same memory,",
        "use duplicate_absorbed with absorbed_into so its derivation count moves there.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .describe("The peer the conclusion is about."),
        conclusion_id: z.string().describe("The conclusion to retire."),
        category: z
          .enum([
            "duplicate_absorbed",
            "superseded",
            "contradicted",
            "misderived",
            "out_of_scope",
            "transient",
            "low_value",
          ])
          .describe(
            "duplicate_absorbed: folded into another conclusion (requires absorbed_into). superseded: was true, reality changed. contradicted: was never true. misderived: extraction defect. out_of_scope: valid but belongs elsewhere, carrier verified. transient: never durable. low_value: not worth carrying.",
          ),
        reason: z
          .string()
          .describe(
            "Specific justification, naming what carries the memory now when something does.",
          ),
        absorbed_into: z
          .string()
          .optional()
          .describe(
            "Required for duplicate_absorbed: the surviving conclusion, which inherits this one's derivation count.",
          ),
        agent_trace_id: z.string().describe("Agent run or trace identifier."),
        agent_model: z
          .string()
          .describe("Exact model identifier making the decision."),
      },
    },
    async ({
      workspace_id,
      peer_id,
      target_peer_id,
      conclusion_id,
      category,
      reason,
      absorbed_into,
      agent_trace_id,
      agent_model,
    }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = peer.conclusionsOf(target_peer_id);
        await scope.delete(conclusion_id, {
          category,
          reason,
          absorbed_into,
          entry_origin: "explicit_agent",
          agent_trace_id,
          agent_model,
        });
        return textResult("Conclusion retired successfully");
      } catch (e) {
        return errorResult(
          `Failed to retire conclusion: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── conclusion_lineage ──────────────────────────────────────────────
  server.registerTool(
    "conclusion_lineage",
    {
      description: [
        "Get one conclusion's full ledger, whether it is live or retired.",
        "Returns how it was admitted, every prior formulation it was rewritten from,",
        "everything it absorbed with the count each brought, and why it was removed.",
        "This is the only way to read a retired conclusion; search never returns one.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .describe("The peer the conclusion is about."),
        conclusion_id: z.string().describe("The conclusion to trace."),
      },
    },
    async ({ workspace_id, peer_id, target_peer_id, conclusion_id }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = peer.conclusionsOf(target_peer_id);
        const lineage = await scope.lineage(conclusion_id);
        return textResult({
          id: lineage.id,
          content: lineage.content,
          times_derived: lineage.timesDerived,
          created_at: lineage.createdAt,
          deleted_at: lineage.deletedAt,
          admission: lineage.admission,
          removal: lineage.removal,
          admission_history: lineage.admissionHistory,
          absorbed: lineage.absorbed,
        });
      } catch (e) {
        return errorResult(
          `Failed to read conclusion lineage: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}
