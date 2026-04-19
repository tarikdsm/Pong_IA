export class PongEngineError extends Error {}

export class InvalidGameStateError extends PongEngineError {}

export class InvalidActionError extends PongEngineError {}

export class MissingRngError extends PongEngineError {}
