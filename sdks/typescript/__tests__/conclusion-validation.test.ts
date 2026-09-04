import { describe, expect, test } from 'bun:test'

import {
  type ConclusionCreateParams,
  ConclusionScope,
  MAX_CONCLUSION_CHARS,
} from '../src'

function operatorConclusion(content: string): ConclusionCreateParams {
  return {
    content,
    action: 'create',
    reasonForEntry: 'test',
    searchQuery: 'test',
    searchedConclusionIds: [],
    sourceToolCallId: 'test',
    entryOrigin: 'operator_sdk',
    agentTraceId: 'test',
    agentModel: 'test',
  }
}

describe('conclusion content validation', () => {
  test('rejects more than 800 Unicode characters before HTTP', async () => {
    const http = {
      post: async () => {
        throw new Error('HTTP should not be called')
      },
    }
    const scope = new ConclusionScope(
      http as never,
      'workspace',
      'observer',
      'observed'
    )

    await expect(
      scope.create(operatorConclusion('😀'.repeat(MAX_CONCLUSION_CHARS + 1)))
    ).rejects.toThrow('conclusion content cannot exceed 800 characters')
  })
})
