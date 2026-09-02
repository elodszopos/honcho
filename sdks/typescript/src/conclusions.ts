import { API_VERSION } from './api-version'
import type { HonchoHTTPClient } from './http/client'
import { Page } from './pagination'
import type { Session } from './session'
import type {
  ConclusionLevel,
  ConclusionResponse,
  PageResponse,
  RepresentationOptions,
  RepresentationResponse,
} from './types/api'
import { normalizeSearchQuery, RepresentationOptionsSchema } from './validation'

/**
 * Filter keys that define a conclusion scope (the observer/observed peer pair).
 * They are set from the scope itself, so a caller must not pass them in `filters`.
 */
const SCOPE_RESERVED_KEYS = [
  'observer',
  'observed',
  'observer_id',
  'observed_id',
]

/**
 * Throw if `filters` contains keys managed by the conclusion scope.
 *
 * The observer/observed peer pair (and, on `list`, the session) is fixed by the
 * scope, so letting a user filter override it would silently return data from a
 * different scope than requested. Fail loud instead.
 */
function rejectReservedFilterKeys(
  filters: Record<string, unknown> | undefined,
  reserved: string[]
): void {
  if (!filters) return
  const clash = reserved.filter((k) => k in filters).sort()
  if (clash.length > 0) {
    let guidance =
      'Choose the peer pair via peer.conclusions / peer.conclusionsOf(target)'
    if (reserved.includes('session') || reserved.includes('session_id')) {
      guidance += '; use the session option to filter by session'
    }
    throw new Error(
      `Filter key(s) ${clash.join(', ')} are managed by this conclusion scope ` +
        `and cannot be passed in filters. ${guidance}.`
    )
  }
}

/**
 * Parameters for creating a conclusion.
 */
export interface ConclusionCreateParams {
  /** The conclusion content/text */
  content: string
  /** The session this conclusion relates to (ID string or Session object) */
  sessionId?: string | Session
  level?: ConclusionLevel
  action: 'create' | 'enrich'
  targetId?: string
  reasonForEntry: string
  searchQuery: string
  searchedConclusionIds: string[]
  sourceMessageIds?: number[]
  sourceToolCallId?: string
  entryOrigin:
    | 'deriver_agent'
    | 'dreamer_agent'
    | 'explicit_agent'
    | 'operator_cli'
    | 'operator_sdk'
    | 'operator_import'
  agentTraceId: string
  agentModel: string
  timesDerived?: number
  sourceIds?: string[]
  premises?: string[]
  sources?: string[]
  patternType?:
    | 'preference'
    | 'behavior'
    | 'personality'
    | 'tendency'
    | 'correlation'
  confidence?: 'high' | 'medium' | 'low'
}

/**
 * A conclusion from Honcho's reasoning system.
 *
 * Conclusions are facts derived from messages that help build a representation
 * of a peer.
 */
export class Conclusion {
  readonly id: string
  readonly content: string
  readonly observerId: string
  readonly observedId: string
  readonly sessionId: string | null
  /**
   * Reasoning level: 'explicit' conclusions are extracted directly from
   * messages; 'deductive'/'inductive'/'contradiction' are derived during
   * dreaming.
   */
  readonly level: ConclusionLevel
  readonly admission: Record<string, unknown>
  readonly admissionHistory: Array<Record<string, unknown>>
  readonly createdAt: string

  constructor(
    id: string,
    content: string,
    observerId: string,
    observedId: string,
    sessionId: string | null,
    createdAt: string,
    level: ConclusionLevel = 'explicit',
    admission: Record<string, unknown> = {},
    admissionHistory: Array<Record<string, unknown>> = []
  ) {
    this.id = id
    this.content = content
    this.observerId = observerId
    this.observedId = observedId
    this.sessionId = sessionId
    this.level = level
    this.admission = admission
    this.admissionHistory = admissionHistory
    this.createdAt = createdAt
  }

  static fromApiResponse(data: ConclusionResponse): Conclusion {
    return new Conclusion(
      data.id,
      data.content,
      data.observer_id,
      data.observed_id,
      data.session_id,
      data.created_at,
      data.level,
      data.admission,
      data.admission_history
    )
  }

  toString(): string {
    const truncatedContent =
      this.content.length > 50
        ? `${this.content.slice(0, 50)}...`
        : this.content
    return `Conclusion(id='${this.id}', content='${truncatedContent}')`
  }
}

/**
 * Scoped access to conclusions for a specific observer/observed relationship.
 */
export class ConclusionScope {
  private _http: HonchoHTTPClient
  private _ensureWorkspace: () => Promise<void>
  readonly workspaceId: string
  readonly observer: string
  readonly observed: string

  constructor(
    http: HonchoHTTPClient,
    workspaceId: string,
    observer: string,
    observed: string,
    ensureWorkspace: () => Promise<void> = async () => undefined
  ) {
    this._http = http
    this.workspaceId = workspaceId
    this.observer = observer
    this.observed = observed
    this._ensureWorkspace = ensureWorkspace
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _list(params: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
    reverse?: boolean
  }): Promise<PageResponse<ConclusionResponse>> {
    await this._ensureWorkspace()
    return this._http.post<PageResponse<ConclusionResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/list`,
      {
        body: { filters: params.filters },
        query: {
          page: params.page,
          size: params.size,
          reverse: params.reverse ? 'true' : undefined,
        },
      }
    )
  }

  private async _query(params: {
    query: string
    top_k?: number
    distance?: number
    filters?: Record<string, unknown>
  }): Promise<ConclusionResponse[]> {
    await this._ensureWorkspace()
    return this._http.post<ConclusionResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/query`,
      { body: params }
    )
  }

  private async _create(params: {
    conclusions: Array<Record<string, unknown>>
  }): Promise<ConclusionResponse[]> {
    await this._ensureWorkspace()
    return this._http.post<ConclusionResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions`,
      { body: params }
    )
  }

  private async _delete(conclusionId: string): Promise<void> {
    await this._ensureWorkspace()
    await this._http.delete(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/${conclusionId}`
    )
  }

  private async _getRepresentation(
    peerId: string,
    params: {
      target?: string
      search_query?: string
      search_top_k?: number
      search_max_distance?: number
      include_most_frequent?: boolean
      max_conclusions?: number
    }
  ): Promise<RepresentationResponse> {
    await this._ensureWorkspace()
    return this._http.post<RepresentationResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${peerId}/representation`,
      { body: params }
    )
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  /**
   * List conclusions in this scope.
   *
   * @param options - Optional configuration for the list request
   * @param options.page - Page number (1-indexed, default: 1)
   * @param options.size - Number of items per page (default: 50)
   * @param options.session - Optional session (ID string or Session object) to filter by
   * @param options.filters - Optional additional filter criteria, merged with
   *   this scope's observer/observed (and session, if given). Supports the same
   *   operators as other list endpoints — e.g. `{ level: 'explicit' }` to get
   *   only conclusions extracted directly from messages (i.e. not derived during
   *   dreaming). See
   *   https://honcho.dev/docs/v3/documentation/features/advanced/using-filters
   * @returns Promise resolving to a Page of Conclusion objects
   */
  async list(options?: {
    page?: number
    size?: number
    session?: string | Session
    filters?: Record<string, unknown>
    reverse?: boolean
  }): Promise<Page<Conclusion, ConclusionResponse>> {
    rejectReservedFilterKeys(options?.filters, [
      ...SCOPE_RESERVED_KEYS,
      'session',
      'session_id',
    ])
    const resolvedSessionId = options?.session
      ? typeof options.session === 'string'
        ? options.session
        : options.session.id
      : undefined
    const filters: Record<string, unknown> = {
      observer_id: this.observer,
      observed_id: this.observed,
      ...(resolvedSessionId ? { session_id: resolvedSessionId } : {}),
      ...options?.filters,
    }
    const reverse = options?.reverse

    const response = await this._list({
      filters,
      page: options?.page ?? 1,
      size: options?.size ?? 50,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<ConclusionResponse>> => {
      return this._list({ filters, page, size, reverse })
    }

    return new Page(
      response,
      (item) => Conclusion.fromApiResponse(item),
      fetchNextPage
    )
  }

  /**
   * Semantic search for conclusions in this scope.
   *
   * @param query - The search query string
   * @param topK - Maximum number of results to return (default: 10)
   * @param distance - Maximum cosine distance threshold (0.0-1.0)
   * @param filters - Optional additional filter criteria, merged with this
   *   scope's observer/observed. Supports the same operators as the list
   *   endpoint — e.g. `{ level: 'deductive' }` to search only conclusions
   *   derived during dreaming. See
   *   https://honcho.dev/docs/v3/documentation/features/advanced/using-filters
   */
  async query(
    query: string,
    topK: number = 10,
    distance?: number,
    filters?: Record<string, unknown>
  ): Promise<Conclusion[]> {
    rejectReservedFilterKeys(filters, SCOPE_RESERVED_KEYS)
    const response = await this._query({
      query,
      top_k: topK,
      distance,
      filters: {
        observer_id: this.observer,
        observed_id: this.observed,
        ...filters,
      },
    })

    return (response ?? []).map((item) => Conclusion.fromApiResponse(item))
  }

  /**
   * Delete a conclusion by ID.
   */
  async delete(conclusionId: string): Promise<void> {
    await this._delete(conclusionId)
  }

  /** Create or enrich conclusions through the public admission API. */
  async create(
    conclusions: ConclusionCreateParams | ConclusionCreateParams[]
  ): Promise<Conclusion[]> {
    const conclusionArray = Array.isArray(conclusions)
      ? conclusions
      : [conclusions]
    const requestConclusions: Array<Record<string, unknown>> = []

    for (const obs of conclusionArray) {
      if (obs.action === 'create' && obs.targetId !== undefined) {
        throw new Error('create decisions cannot set targetId')
      }
      if (obs.action === 'enrich') {
        if (obs.targetId === undefined) {
          throw new Error('enrich decisions require targetId')
        }
        if (!obs.searchedConclusionIds.includes(obs.targetId)) {
          throw new Error('targetId must be present in searchedConclusionIds')
        }
      }
      if (
        obs.entryOrigin === 'deriver_agent' &&
        !obs.sourceMessageIds?.length
      ) {
        throw new Error('deriver_agent conclusions require sourceMessageIds')
      }
      if (obs.entryOrigin === 'dreamer_agent' && !obs.sourceIds?.length) {
        throw new Error('dreamer_agent conclusions require sourceIds')
      }
      if (
        obs.entryOrigin === 'explicit_agent' &&
        !(obs.sourceMessageIds?.length || obs.sourceToolCallId)
      ) {
        throw new Error(
          'explicit_agent conclusions require sourceMessageIds or sourceToolCallId'
        )
      }
      if (obs.entryOrigin.startsWith('operator_') && !obs.sourceToolCallId) {
        throw new Error('operator conclusions require sourceToolCallId')
      }

      const sessionId =
        obs.sessionId === undefined
          ? null
          : typeof obs.sessionId === 'string'
            ? obs.sessionId
            : obs.sessionId.id
      requestConclusions.push({
        content: obs.content,
        session_id: sessionId,
        observer_id: this.observer,
        observed_id: this.observed,
        level: obs.level ?? 'explicit',
        action: obs.action,
        target_id: obs.targetId,
        reason_for_entry: obs.reasonForEntry,
        search_query: obs.searchQuery,
        searched_conclusion_ids: obs.searchedConclusionIds,
        source_message_ids: obs.sourceMessageIds ?? [],
        source_tool_call_id: obs.sourceToolCallId,
        entry_origin: obs.entryOrigin,
        agent_trace_id: obs.agentTraceId,
        agent_model: obs.agentModel,
        times_derived: obs.timesDerived,
        source_ids: obs.sourceIds,
        premises: obs.premises,
        sources: obs.sources,
        pattern_type: obs.patternType,
        confidence: obs.confidence,
      })
    }

    const response = await this._create({ conclusions: requestConclusions })
    return (response ?? []).map((item) => Conclusion.fromApiResponse(item))
  }

  /**
   * Get the computed representation for this scope.
   */
  async representation(options?: RepresentationOptions): Promise<string> {
    const searchQuery = normalizeSearchQuery(options?.searchQuery)
    const validatedOptions = RepresentationOptionsSchema.parse({
      searchQuery,
      searchTopK: options?.searchTopK,
      searchMaxDistance: options?.searchMaxDistance,
      includeMostFrequent: options?.includeMostFrequent,
      maxConclusions: options?.maxConclusions,
    })

    const response = await this._getRepresentation(this.observer, {
      target: this.observed,
      search_query: searchQuery,
      search_top_k: validatedOptions.searchTopK,
      search_max_distance: validatedOptions.searchMaxDistance,
      include_most_frequent: validatedOptions.includeMostFrequent,
      max_conclusions: validatedOptions.maxConclusions,
    })
    return response.representation
  }

  toString(): string {
    return `ConclusionScope(workspaceId='${this.workspaceId}', observer='${this.observer}', observed='${this.observed}')`
  }
}
