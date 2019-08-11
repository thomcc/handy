//! The "advanced usage" module.
//!
//! Allows things like manual handle allocation, inspection of a handlemap's
//! slots, etc.

use crate::{Handle, HandleMap};

/// An allocator for handle values. This is useful if you need direct control
/// over handle storage, for example if you want use this library just to
/// provide abstract generational indices which can be used in multiple backing
/// arrays.
///
/// Besides the name, the only difference between `HandleAlloc` and a
/// `HandleMap<()>` is that `HandleAlloc` completely ignores the value in the
/// `meta` field of the handle, which the `HandleMap` uses for the map id.
///
/// The primary use case for this is if you want to use the handles to index
/// into separate storage, for example if you have more than one Vec<T> you'd
/// like to use with them.
#[derive(Debug, Clone)]
pub struct HandleAlloc {
    // Seems like it should be the other way around right? But this way we can
    // store entries inline on HandleMap, and have no real downsides on this
    // end.
    inner: HandleMap<()>,
}

impl Default for HandleAlloc {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl HandleAlloc {
    /// Construct a new handle allocator.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: HandleMap::new_with_map_id(0),
        }
    }

    /// Returns a new allocator with the requested capacity.
    #[inline]
    pub fn with_capacity(c: usize) -> Self {
        let mut m = Self::new();
        m.reserve(c);
        m
    }

    /// Returns the capacity of this handle allocator, which is equivalent to
    /// the maximum possible `index` value a handle may currently have at the
    /// moment.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns the number of slots that are currently occupied.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if there are no currently occupied slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Deallocate all current slots. Note that this changes all of their
    /// generations, and also defragments the free-list so that lower items are
    /// once again preferred.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Reserve space for `sz` additional handles.
    pub fn reserve(&mut self, sz: usize) {
        self.inner.reserve(sz);
    }

    /// Allocate a handle. This is O(1) unless we must reallocate the backing
    /// memory.
    ///
    /// In general, the algorithm attempts to return handles with lower indices
    /// preferentially, but this is not guaranteed. For example, the current
    /// implementation of the allocator uses a free list, and dealloc adds items
    /// to the front of that list.
    #[inline]
    pub fn alloc(&mut self) -> Handle {
        self.inner.insert(())
    }

    /// Deallocate a handle, freeing its index to be returned from a subsequent
    /// call to alloc.
    #[inline]
    pub fn dealloc(&mut self, h: Handle) -> bool {
        if let Some(i) = self.test_handle(h) {
            self.inner.raw_remove(i);
            true
        } else {
            false
        }
    }

    /// If `index` refers to an occupied entry, return an index to it.
    /// Otherwise, return None.
    #[inline]
    pub fn handle_for_index(&self, index: usize) -> Option<Handle> {
        self.inner.handle_for_index(index)
    }

    /// If `h` is a valid handle, get it's index value. Otherwise return None.
    #[inline]
    pub fn test_handle(&self, h: Handle) -> Option<usize> {
        if self.contains(h) {
            Some(h.index())
        } else {
            None
        }
    }

    /// Returns true if `h` is valid handle that refers to an occupied slot in
    /// this allocator.
    #[inline]
    pub fn contains(&self, h: Handle) -> bool {
        // Equivalent to making a fake handle with the correct map id, and
        // calling validate_handle with it.
        let i = h.index() as usize;
        let gen = h.generation();
        if i >= self.inner.entries.len() || (gen & 1) != 0 || self.inner.entries[i].gen != gen {
            super::unlikely_hint();
            false
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_allocator() {
        let mut a = HandleAlloc::with_capacity(10);
        for i in 0..10 {
            let v = a.alloc();
            assert_eq!(v.generation(), 2);
            assert_eq!(v.meta(), 0);
            assert_eq!(v.index(), i);
        }
        assert_eq!(a.capacity(), 10);
        for i in 0..10 {
            assert_eq!(a.handle_for_index(i), Some(Handle::from_raw_parts(i, 2, 0)));
        }

        for i in (0..10).rev() {
            assert_eq!(a.handle_for_index(i), Some(Handle::from_raw_parts(i, 2, 0)));
            assert!(!a.dealloc(Handle::from_raw_parts(i, 1, 0))); // bad gen
            assert!(!a.dealloc(Handle::from_raw_parts(i, 0, 0))); // bad gen but even
            assert_eq!(a.test_handle(Handle::from_raw_parts(i, 2, 0)), Some(i));

            // Should remove
            assert_eq!(a.dealloc(Handle::from_raw_parts(i, 2, 0xff)), true);
            // Just removed
            assert_eq!(a.dealloc(Handle::from_raw_parts(i, 2, 0xff)), false);
            // refers to the right slot and gen, but not occupied.
            assert_eq!(a.dealloc(Handle::from_raw_parts(i, 3, 0xff)), false);
        }
        assert!(a.is_empty());
    }
}
