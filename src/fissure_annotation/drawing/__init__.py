"""Everything that puts pixels on an image, as pure PIL functions.

- Nothing here knows about slices, views or frames. Callers convert to image pixels first.
- That keeps image code reviewable in isolation, and testable without a CT volume.
"""
