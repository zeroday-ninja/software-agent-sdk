# Changes this repository did not want

- **A breaking contract change without the client.** The generated TypeScript
  client is part of the contract, not a follow-up.
- **Freezing a resolved path in a module constant.** A process that sets its
  persistence directory after import then reads a different file than it
  writes.
- **An exception handler that returns a default.** If a document exists but
  cannot be read, that is not the same as it never having been written, and a
  caller needs to be able to tell.
- **Exporting an internal to make it testable.** Reach the behaviour through
  the public path.
