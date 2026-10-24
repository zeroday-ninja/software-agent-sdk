# How code and prose are written here

## Comments

Why, never what. The module docstring states the reason the module exists and
the decision that shaped it; the function docstring states what a caller can
rely on. A comment that restates the next line is deleted.

## Public surface

Every exported name is part of a contract someone else depends on. `__all__`
is maintained deliberately. A helper that exists only for a test does not get
exported; the test reaches the behaviour through the public path instead.

## Types

Strict. `str | None` where None means something, and the docstring says what it
means — "missing, unreadable, empty and whitespace-only alike" is a real
distinction a caller has to be told about.

## Tests

Properties over transcripts. For anything mergeable or ordered, the test states
the law: the union keeps everything, merging in either direction agrees,
merging twice changes nothing.
