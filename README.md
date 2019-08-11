# `handy`: Handles are handy.

[![Docs](https://docs.rs/handy/badge.svg)](https://docs.rs/handy) [![CircleCI](https://circleci.com/gh/thomcc/handy.svg?style=svg)](https://circleci.com/gh/thomcc/handy) [![codecov](https://codecov.io/gh/thomcc/handy/branch/master/graph/badge.svg)](https://codecov.io/gh/thomcc/handy)

`handy` provides handles and handle maps. This is a fairly useful data
structure for rust code, since it can help you work around borrow checker
issues.

Essentially, `Handle` and `HandleMap` are a more robust version of the
pattern where instead of storing a reference to a &T directly, you instead
store a `usize` which indicates where it is in some `Vec`. I claim they're
more robust because:

- They can detect if you try to use a handle in a map other than the one
  that provided it.

- If you remove an item from the HandleMap, the handle map won't let you use
  the stale handle to get whatever value happens to be in that slot at the
  time.

## Similar crates

There are a whole bunch.

- `slotmap`: Same idea as this, but it requires `T: Copy` (there's a way
  around this but it's a pain IMO). Has a system for defining handles for
  use in specific maps, but can't detect if you use a key from one map in
  another, if the maps are the same type. It also has a bunch of other maps
  for different performance cases but honestly the limitation of `T: Copy`
  has prevented me from digging too deeply.

- `slab`: Also the same idea but you might not realize it from the docs. It
  can't detect use with the wrong map or use after the item is removed and
  another occupies the same spot.

- `ffi_support`'s `HandleMap`: I wrote this one. It's very similar, but with
  some different tradeoffs, and essentially different code. Also, this
  library doesn't bring in as many heavyweight dependencies, has more
  features, and isn't focused on use inside the FFI.

- Unlike any of them, we're usable in no_std situations (we do link with
  `extern crate alloc`, of course).

## License

TLDR: MIT / Apache2 like every other Rust library.

This code shares common legacy with some of the types in
https://crates.io/crates/ffi-support. After all, I wrote that library too. I
also stole the test code more or less directly from that crate. Because of that,
this library has the exact same license, down to the copyright assignment to
Mozilla.

In practice that doesn't matter, since who cares, they're both MIT / Apache2,
but I figured I should write it down somewhere.
