import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho } from '../src'
import { createTestClient, requireServer } from './setup'

async function captureChatBodies(
  run: () => Promise<unknown>
): Promise<Record<string, unknown>[]> {
  const bodies: Record<string, unknown>[] = []
  const originalFetch = globalThis.fetch
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).endsWith('/chat') && typeof init?.body === 'string') {
      bodies.push(JSON.parse(init.body))
    }
    return originalFetch(input, init)
  }) as typeof fetch
  try {
    await run()
  } finally {
    globalThis.fetch = originalFetch
  }
  return bodies
}

describe('Peer chat automatic option', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('automatic-chat')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  test('sends every option in the server wire shape', async () => {
    const peer = await client.peer('automatic-chat-peer')
    let response: unknown

    const bodies = await captureChatBodies(async () => {
      response = await peer.chat('What did we decide about the reminder job?', {
        automatic: {
          searchText: 'reminder job | fire at 9',
          excludeConclusionIds: ['c-1', 'c-2'],
          excludeSessionId: 'thread-1',
          excerptLimit: 3,
          maxAnswerChars: 600,
        },
      })
    })

    expect(response === null || typeof response === 'string').toBe(true)
    expect(bodies).toHaveLength(1)
    expect(bodies[0].stream).toBe(false)
    expect(bodies[0].automatic).toEqual({
      search_text: 'reminder job | fire at 9',
      exclude_conclusion_ids: ['c-1', 'c-2'],
      exclude_session_id: 'thread-1',
      excerpt_limit: 3,
      max_answer_chars: 600,
    })
  })

  test('sends only the options given', async () => {
    const peer = await client.peer('automatic-chat-minimal-peer')

    const bodies = await captureChatBodies(() =>
      peer.chat('What did we decide?', {
        automatic: { searchText: 'reminder job' },
      })
    )

    expect(bodies[0].automatic).toEqual({ search_text: 'reminder job' })
  })

  test('omits the option when none is given', async () => {
    const peer = await client.peer('automatic-chat-plain-peer')

    const bodies = await captureChatBodies(() => peer.chat('What did we decide?'))

    expect(bodies).toHaveLength(1)
    expect('automatic' in bodies[0]).toBe(false)
  })

  test.each([
    { searchText: '' },
    { searchText: 'reminder job', excerptLimit: 21 },
    { searchText: 'reminder job', maxAnswerChars: 99 },
    { searchText: 'reminder job', excludeConclusionIds: Array(1001).fill('c') },
  ])('refuses %p before any request', async (automatic) => {
    const peer = await client.peer('automatic-chat-invalid-peer')

    const bodies = await captureChatBodies(async () => {
      await expect(peer.chat('What did we decide?', { automatic })).rejects.toThrow()
    })

    expect(bodies).toHaveLength(0)
  })
})
