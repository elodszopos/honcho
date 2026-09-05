import { z } from "zod";

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── list_conclusions ────────────────────────────────────────────────
  server.registerTool(
    "list_conclusions",
    {
      description: [
        "List conclusions (facts and observations) that Honcho has derived about a peer (paginated).",
        "Use this to see what Honcho has learned. If no target is given, returns self-conclusions.",
        "Returns conclusion objects with pagination metadata.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: list conclusions about this target. Omit for self-conclusions.",
          ),
      },
    },
    async ({ peer_id, target_peer_id }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const scope = target_peer_id
          ? peer.conclusionsOf(target_peer_id)
          : peer.conclusions;
        const page = await scope.list();
        return textResult({
          conclusions: page.items.map((c) => ({
            id: c.id,
            content: c.content,
            observer_id: c.observerId,
            observed_id: c.observedId,
            session_id: c.sessionId,
            times_derived: c.timesDerived,
            created_at: c.createdAt,
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
      },
    },
    async ({ peer_id, query, target_peer_id, top_k }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const scope = target_peer_id
          ? peer.conclusionsOf(target_peer_id)
          : peer.conclusions;
        const conclusions = await scope.query(query, top_k);
        return textResult(
          conclusions.map((c) => ({
            id: c.id,
            content: c.content,
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
    async ({ peer_id, target_peer_id, conclusions, agent_trace_id, agent_model, session_id }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
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
        }));
        await scope.create(params);
        return textResult(
          `Created ${conclusions.length} conclusion${conclusions.length === 1 ? "" : "s"} successfully`,
        );
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
        "Delete a specific conclusion by ID.",
        "Use this to remove incorrect or outdated knowledge.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .describe("The peer the conclusion is about."),
        conclusion_id: z.string().describe("The conclusion to delete."),
      },
    },
    async ({ peer_id, target_peer_id, conclusion_id }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const scope = peer.conclusionsOf(target_peer_id);
        await scope.delete(conclusion_id);
        return textResult("Conclusion deleted successfully");
      } catch (e) {
        return errorResult(
          `Failed to delete conclusion: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}
