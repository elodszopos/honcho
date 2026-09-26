import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho } from '../src'
import type { ConclusionCreateParams } from '../src'
import { createTestClient, requireServer } from './setup'

function operatorConclusion(
  content: string,
  sessionId?: ConclusionCreateParams['sessionId']
): ConclusionCreateParams {
  return {
    content,
    sessionId,
    action: 'create',
    reasonForEntry: 'Operator explicitly requested durable storage.',
    searchQuery: content,
    searchedConclusionIds: [],
    sourceToolCallId: 'typescript-sdk-test-operator',
    entryOrigin: 'operator_sdk',
    agentTraceId: 'typescript-sdk-test-trace',
    agentModel: 'typescript-sdk-test-model',
  }
}

describe('Conclusion query distance', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('conclusion-distance')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  test('query results carry their distance, closest first', async () => {
    const peer = await client.peer('distance-query-peer', { metadata: {} })
    const session = await client.session('distance-query-session', { metadata: {} })
    await peer.conclusions.create([
      operatorConclusion('User loves Italian cuisine, especially pasta', session),
      operatorConclusion('User runs along the river every morning', session),
      operatorConclusion('User keeps paper receipts in a shoebox', session),
    ])

    const results = await peer.conclusions.query('food preferences', 3)

    expect(results.length).toBeGreaterThan(0)
    const distances = results.map((conclusion) => conclusion.distance)
    for (const distance of distances) {
      expect(typeof distance).toBe('number')
      expect(distance as number).toBeGreaterThanOrEqual(0)
      expect(distance as number).toBeLessThanOrEqual(2)
    }
    const ranked = [...distances].sort((a, b) => (a as number) - (b as number))
    expect(distances).toEqual(ranked)
  })

  test('a per-thought query ranks by best distance', async () => {
    const peer = await client.peer('thought-query-peer', { metadata: {} })
    const session = await client.session('thought-query-session', { metadata: {} })
    await peer.conclusions.create([
      operatorConclusion('User loves Italian cuisine, especially pasta', session),
      operatorConclusion('User runs along the river every morning', session),
    ])

    const results = await peer.conclusions.query(
      'what food do they like?\nand where do they run?',
      2,
      undefined,
      undefined,
      true
    )

    expect(results.length).toBe(2)
    const distances = results.map((conclusion) => conclusion.distance as number)
    expect(distances).toEqual([...distances].sort((a, b) => a - b))
  })

  test('listed conclusions carry no distance', async () => {
    const peer = await client.peer('distance-list-peer', { metadata: {} })
    const session = await client.session('distance-list-session', { metadata: {} })
    await peer.conclusions.create(
      operatorConclusion('User drinks green tea in the afternoon', session)
    )

    const page = await peer.conclusions.list()

    expect(page.items.length).toBeGreaterThan(0)
    for (const conclusion of page.items) {
      expect(conclusion.distance).toBeNull()
    }
  })
})
