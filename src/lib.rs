#![no_std]
#![allow(clippy::let_and_return)]
#![deny(unsafe_code, missing_docs)]
//! # `handy`
//!
//! `handy` provides handles and handle maps. A handle map is a fairly useful
//! data structure for rust code, since it can help you work around borrow
//! checker issues.
//!
//! Essentially, [`Handle`] and [`HandleMap`] are a more robust version of the
//! pattern where instead of storing a reference to a &T directly, you instead
//! store a `usize` which indicates where it is in some `Vec`. I claim they're
//! more robust because:
//!
//! - They can detect if you try to use a handle in a map other than the one
//!   that provided it.
//!
//! - If you remove an item from the HandleMap, the handle map won't let you use
//!   the stale handle to get whatever value happens to be in that index at the
//!   time.
//!
//! ## Usage Example
//!
//! ```
//! # use handy::HandleMap;
//! let mut m = HandleMap::new();
//! let h0 = m.insert(3u32);
//! assert_eq!(m[h0], 3);
//! m.remove(h0);
//! assert_eq!(m.get(h0), None);
//! ```
//!
//! # Similar crates
//!
//! A whole bunch.
//!
//! - `slotmap`: Same idea as this, but it requires `T: Copy` (there's a way
//!   around this but it's a pain IMO). Has a system for defining handles for
//!   use in specific maps, but can't detect if you use a key from one map in
//!   another, if the maps are the same type. It also has a bunch of other maps
//!   for different performance cases but honestly the limitation of `T: Copy`
//!   has prevented me from digging too deeply.
//!
//! - `slab`: Also the same idea but you might not realize it from the docs. It
//!   can't detect use with the wrong map or use after the item is removed and
//!   another occupies the same spot.
//!
//! - `ffi_support`'s `HandleMap`: I wrote this one. It's very similar, but with
//!   some different tradeoffs, and essentially different code. Also, this
//!   library doesn't bring in as many heavyweight dependencies, has more
//!   features, and isn't focused on use inside the FFI.
//!
//! - Unlike any of them, we're usable in no_std situations (we do link with
//!   `extern crate alloc`, of course).

extern crate alloc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, Ordering};
mod halloc;

pub mod typed;

pub use halloc::HandleAlloc;

/// `HandleMap`s are a collection data structure that allow you to reference
/// their members by using a opaque handle.
///
/// In rust code, you often use these handles as a sort of lifetime-less
/// reference. `Handle` is a paper-thin wrapper around a `u64`, so it is `Copy +
/// Send + Sync + Eq + Ord + ...` even if `T` (or even `&T`) wouldn't be,
/// however you need access to the map in order to read the value.
///
/// This is probably starting to sound like `HandleMap` is just a `Vec`, and
/// `Handle` is just `usize`, but unlike `usize`:
///
/// - a `HandleMap` can tell if you try to use a `Handle` from a different map
///   to access one of it's values.
///
/// - a `HandleMap` tracks insertion/removal of the value at each index, will
///   know if you try to use a handle to get a value that was removed, even if
///   another value occupies the same index.
///
/// # Example
/// ```
/// # use handy::HandleMap;
/// let mut m = HandleMap::new();
/// let h0 = m.insert(3usize);
/// assert_eq!(m[h0], 3);
/// m[h0] += 2;
/// assert_eq!(m[h0], 5);
/// let v = m.remove(h0);
/// assert_eq!(v, Some(5));
/// assert_eq!(m.get(h0), None);
/// ```
#[derive(Clone)]
pub struct HandleMap<T> {
    entries: Vec<Entry<T>>,
    len: usize,
    next: Option<u32>,
    end_of_list: Option<u32>,
    id: u16,
}

impl<T> Default for HandleMap<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

static SOURCE_ID: AtomicU16 = AtomicU16::new(1);

impl<T> HandleMap<T> {
    /// Create a new handle map.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let m: HandleMap<u32> = HandleMap::new();
    /// // No allocation is performed by default.
    /// assert_eq!(m.capacity(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::new_with_map_id(SOURCE_ID.fetch_add(1, Ordering::Relaxed))
    }

    #[inline]
    pub(crate) fn new_with_map_id(id: u16) -> Self {
        Self {
            entries: Vec::new(),
            len: 0,
            next: None,
            end_of_list: None,
            id,
        }
    }

    /// Create a new handle map with at least the specified capacity.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let m: HandleMap<u32> = HandleMap::with_capacity(10);
    /// // Note that we don't guarantee the capacity will be exact.
    /// // (though in practice it will so long as the requested
    /// // capacity is >= 8)
    /// assert!(m.capacity() >= 10);
    /// ```
    pub fn with_capacity(c: usize) -> Self {
        let mut a = Self::new();
        if c == 0 {
            return a;
        }
        assert!(c < i32::max_value() as usize);
        a.reserve(c);
        a
    }

    /// Get the number of entries we can hold before reallocation
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let m: HandleMap<u32> = HandleMap::with_capacity(10);
    /// assert!(m.capacity() >= 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of occupied entries.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// assert_eq!(m.len(), 0);
    /// m.insert(10u32);
    /// assert_eq!(m.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if our length is zero
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// assert!(m.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the id of this map, which is used to validate handles.
    ///
    /// This is typically not needed except for debugging and advanced usage.
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.map_id(), h.map_id());
    /// ```
    #[inline]
    pub fn map_id(&self) -> u16 {
        self.id
    }

    /// Set the id of this map, which is used to validate handles (See
    /// [`Handle`] documentation for more details).
    ///
    /// # Warning
    /// Doing so will cause the map to fail to recognize handles that it
    /// previously returned, and probably other problems! You're recommended
    /// against using it unless you know what you're doing!
    #[inline]
    pub fn raw_set_map_id(&mut self, v: u16) {
        self.id = v;
    }

    /// Add a new item, returning a handle to it.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m[h], 10);
    /// ```
    pub fn insert(&mut self, value: T) -> Handle {
        let index = self.get_next();
        let mut e = &mut self.entries[index];

        debug_assert!(e.payload.is_none());
        e.payload = Some(value);
        e.gen = e.gen.wrapping_add(1);

        if e.gen == 0 {
            // Zero generation indicates an invalid handle.
            e.gen = 2;
        }
        self.next = core::mem::replace(&mut e.next, None);
        self.len += 1;
        let res = Handle::from_raw_parts(index, e.gen, self.id);
        #[cfg(test)]
        {
            self.assert_valid();
        }
        res
    }

    /// Remove the value referenced by this handle from the map, returning it.
    ///
    /// If the handle doesn't point to an entry in the map we return None. This
    /// will happen if:
    ///
    /// - The handle comes from a different map.
    /// - The item it referenced has been removed already.
    /// - It appears corrupt in some other way (For example, it's
    ///   `Handle::default()`, or comes from a dubious `Handle::from_raw_*`)
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// // Present:
    /// assert_eq!(m.remove(h), Some(10));
    /// // Not present:
    /// assert_eq!(m.remove(h), None);
    /// ```
    pub fn remove(&mut self, handle: Handle) -> Option<T> {
        self.handle_check_mut(handle)?;
        self.raw_remove(handle.index())
    }

    /// Remove all entries in this handle map.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// m.clear();
    /// assert_eq!(m.len(), 0);
    /// assert_eq!(m.get(h), None);
    /// ```
    pub fn clear(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let update_gen = move |e: &mut Entry<T>| {
            if (e.gen & 1) == 0 {
                e.gen = e.gen.wrapping_add(1);
            } else {
                e.gen = e.gen.wrapping_add(2);
            }
            if e.gen == 0 {
                e.gen = 1;
            }
        };
        for i in 0..(self.entries.len() - 1) {
            update_gen(&mut self.entries[i]);
            self.entries[i].next = Some((i + 1) as u32);
            self.entries[i].payload = None;
        }
        let mut end = self.entries.last_mut().unwrap();
        update_gen(&mut end);
        end.next = None;
        end.payload = None;
        self.next = Some(0);
        self.end_of_list = Some((self.entries.len() - 1) as u32);
        self.len = 0;
        #[cfg(test)]
        {
            self.assert_valid();
        }
    }

    /// Try and get a reference to the item backed by the handle.
    ///
    /// If the handle doesn't point to an entry in the map we return None. This
    /// will happen if:
    ///
    /// - The handle comes from a different map.
    /// - The item it referenced has been removed already.
    /// - It appears corrupt in some other way (For example, it's
    ///   `Handle::default()`, or comes from a dubious `Handle::from_raw_*`)
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.get(h), Some(&10));
    /// m.remove(h);
    /// assert_eq!(m.get(h), None);
    /// ```
    #[inline]
    pub fn get(&self, handle: Handle) -> Option<&T> {
        self.handle_check(handle).and_then(|e| e.payload.as_ref())
    }

    /// Try and get mutable a reference to the item backed by the handle.
    ///
    /// If the handle doesn't point to an entry in the map we return None. This
    /// will happen if:
    ///
    /// - The handle comes from a different map.
    /// - The item it referenced has been removed already.
    /// - It appears corrupt in some other way (For example, it's
    ///   `Handle::default()`, or comes from a dubious `Handle::from_raw_*`)
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// *m.get_mut(h).unwrap() += 1;
    /// assert_eq!(m[h], 11);
    /// // Note: The following is equivalent if you're going to `unwrap` the result of get_mut:
    /// m[h] += 1;
    /// assert_eq!(m[h], 12);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        self.handle_check_mut(handle)
            .and_then(|e| e.payload.as_mut())
    }

    /// Returns true if the handle refers to an item present in this map.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert!(m.contains(h));
    /// m.remove(h);
    /// assert!(!m.contains(h));
    /// ```
    #[inline]
    pub fn contains(&self, h: Handle) -> bool {
        self.get(h).is_some()
    }

    /// Returns true if the handle refers to an item present in this map.
    ///
    /// This is equivalent to [`HandleMap::contains`] but provided for some
    /// compatibility with other Map apis.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert!(m.contains_key(h));
    /// m.remove(h);
    /// assert!(!m.contains_key(h));
    /// ```
    #[inline]
    pub fn contains_key(&self, h: Handle) -> bool {
        self.get(h).is_some()
    }

    /// Search the map for `item`, and if it's found, return a handle to it.
    ///
    /// If more than one value compare as equal to `item`, it's not specified
    /// which we will return.
    ///
    /// Note that this is a naive O(n) search, so if you want this often, you
    /// might want to store the handle as a field on the value.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.find_handle(&10), Some(h));
    /// assert_eq!(m.find_handle(&11), None);
    /// ```
    #[inline]
    pub fn find_handle(&self, item: &T) -> Option<Handle>
    where
        T: PartialEq,
    {
        for (i, e) in self.entries.iter().enumerate() {
            if e.payload.as_ref() == Some(item) {
                return Some(Handle::from_raw_parts(i, e.gen, self.id));
            }
        }
        None
    }

    /// Reserve space for `sz` additional items.
    ///
    /// ## Example
    ///
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// assert_eq!(m.capacity(), 0);
    /// m.reserve(10);
    /// assert!(m.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, sz: usize) {
        self.grow(self.len() + sz);
    }

    /// Get an iterator over every occupied slot of this map.
    ///
    /// See also `iter_with_handles` if you want the handles during
    /// iteration.
    ///
    /// ## Example
    ///
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// m.insert(10u32);
    /// assert_eq!(*m.iter().next().unwrap(), 10);
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a {
        self.entries.iter().filter_map(|e| e.payload.as_ref())
    }

    /// Get a mut iterator over every occupied slot of this map.
    ///
    /// See also `iter_mut_with_handles` if you want the handles during
    /// iteration.
    ///
    /// ## Example
    ///
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// for v in m.iter_mut() {
    ///     *v += 1;
    /// }
    /// assert_eq!(m[h], 11);
    /// ```
    #[inline]
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.entries.iter_mut().filter_map(|e| e.payload.as_mut())
    }

    /// Get an iterator over every occupied slot of this map, as well as a
    /// handle which can be used to fetch them later.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// # let m: HandleMap<u32> = HandleMap::new();
    /// for (h, v) in m.iter_with_handles() {
    ///     println!("{:?} => {}", h, v);
    /// }
    /// ```
    #[inline]
    pub fn iter_with_handles<'a>(&'a self) -> impl Iterator<Item = (Handle, &'a T)> + 'a {
        self.entries.iter().enumerate().filter_map(move |(i, e)| {
            e.payload
                .as_ref()
                .map(|p| (Handle::from_raw_parts(i, e.gen, self.id), p))
        })
    }

    /// Get a mut iterator over every occupied slot of this map, as well as a
    /// handle which can be used to fetch them later.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// # let mut m = HandleMap::<u32>::new();
    /// for (h, v) in m.iter_mut_with_handles() {
    ///     *v += 1;
    ///     println!("{:?}", h);
    /// }
    /// ```
    #[inline]
    pub fn iter_mut_with_handles<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Handle, &'a mut T)> + 'a {
        let id = self.id;
        self.entries
            .iter_mut()
            .enumerate()
            .filter_map(move |(i, e)| {
                let gen = e.gen;
                e.payload
                    .as_mut()
                    .map(|p| (Handle::from_raw_parts(i, gen, id), p))
            })
    }

    /// If `index` refers to an occupied entry, return a `Handle` to it.
    /// Otherwise, return None. This is a low level API that shouldn't be needed
    /// for typical use.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.handle_for_index(h.index()), Some(h));
    /// ```
    #[inline]
    pub fn handle_for_index(&self, index: usize) -> Option<Handle> {
        let e = self.entries.get(index)?;
        if e.payload.is_some() {
            debug_assert!((e.gen & 1) == 0 && (e.gen != 0));
            Some(Handle::from_raw_parts(index, e.gen, self.id))
        } else {
            None
        }
    }

    #[inline]
    fn handle_check(&self, handle: Handle) -> Option<&Entry<T>> {
        if handle.meta() != self.id {
            unlikely_hint();
            return None;
        }
        let i = handle.index();
        if i >= self.entries.len() {
            unlikely_hint();
            return None;
        }
        let e = &self.entries[i];
        let gen = handle.generation();
        if e.gen != gen || (gen & 1) != 0 {
            unlikely_hint();
            None
        } else {
            Some(e)
        }
    }

    #[inline]
    fn handle_check_mut(&mut self, handle: Handle) -> Option<&mut Entry<T>> {
        if handle.meta() != self.id {
            unlikely_hint();
            return None;
        }
        let i = handle.index();
        if i >= self.entries.len() {
            unlikely_hint();
            return None;
        }
        let e = &mut self.entries[i];
        let gen = handle.generation();
        if e.gen != gen || (gen & 1) != 0 {
            unlikely_hint();
            None
        } else {
            Some(e)
        }
    }

    #[inline]
    fn get_next(&mut self) -> usize {
        if let Some(n) = self.next {
            n as usize
        } else {
            let n = self.grow_for_insert();
            debug_assert!(self.next == Some(n as u32));
            n
        }
    }

    #[cold]
    fn grow_for_insert(&mut self) -> usize {
        self.grow(self.capacity() + 1).expect("bug")
    }

    // note: returns `self.next` unwrapped.
    fn grow(&mut self, need: usize) -> Option<usize> {
        if need <= self.capacity() {
            return self.next.map(|u| u as usize);
        }
        let cap = (self.capacity() * 2).max(need).max(8);
        assert!(cap <= i32::max_value() as usize, "Capacity overflow");

        self.entries.reserve(cap - self.entries.len());

        let current_cap = self.capacity();
        self.entries.extend((current_cap..(cap - 1)).map(|i| Entry {
            next: Some((i + 1) as u32),
            payload: None,
            gen: 1,
        }));

        self.entries.push(Entry {
            next: None,
            payload: None,
            gen: 1,
        });
        if self.next.is_none() {
            self.next = Some(current_cap as u32);
            self.end_of_list = Some((self.entries.len() - 1) as u32);
        } else {
            let end = self.end_of_list.unwrap();
            let ee = &mut self.entries[end as usize];
            debug_assert!(ee.payload.is_none());
            ee.next = Some(current_cap as u32);
            self.end_of_list = Some((self.entries.len() - 1) as u32);
        }
        #[cfg(test)]
        {
            self.assert_valid();
        }
        Some(current_cap as usize)
    }

    #[cfg(test)]
    #[allow(clippy::cognitive_complexity)]
    fn assert_valid(&self) {
        if self.entries.is_empty() {
            return;
        }

        assert!(self.len() <= self.capacity());
        assert!(
            self.capacity() <= i32::max_value() as usize,
            "Entries too large"
        );

        if self.len() == self.capacity() {
            assert!(self.next.is_none());
        } else {
            assert!(self.next.is_some());
        }

        let number_of_ends = self
            .entries
            .iter()
            .filter(|e| e.next.is_none() && e.payload.is_none())
            .count();
        if self.capacity() != 0 {
            let end = self.end_of_list.expect("Should have end") as usize;
            assert_eq!(self.entries[end].next, None);
            if self.capacity() == self.len() {
                assert!(self.entries[end].payload.is_some());
                assert_eq!(number_of_ends, 0);
            } else {
                assert!(self.entries[end].payload.is_none());
                assert_eq!(number_of_ends, 1);
            }
        } else {
            assert_eq!(number_of_ends, 0);
        }
        if self.next.is_none() {
            assert!(self.entries[self.end_of_list.unwrap() as usize]
                .payload
                .is_some());
        }
        // Check that the free list hits every unoccupied item.
        // The tuple is: `(should_be_in_free_list, is_in_free_list)`.
        let mut free_indices = alloc::vec![(false, false); self.capacity()];
        for (i, e) in self.entries.iter().enumerate() {
            if e.payload.is_none() {
                free_indices[i].0 = true;
            } else {
                assert!(e.next.is_none(), "occupied slot in free list");
            }
        }

        let mut next = self.next;
        while let Some(ni) = next {
            let ni = ni as usize;

            assert!(
                ni <= free_indices.len(),
                "Free list contains out of bounds index!"
            );

            assert!(
                free_indices[ni].0,
                "Free list has an index that shouldn't be free! {}",
                ni
            );

            assert!(
                !free_indices[ni].1,
                "Free list hit an index ({}) more than once! Cycle detected!",
                ni
            );

            free_indices[ni].1 = true;

            assert!(self.entries[ni].payload.is_none());
            next = self.entries[ni].next;
            if next.is_none() {
                assert_eq!(Some(ni as u32), self.end_of_list);
            }
        }

        let mut occupied_count = 0;
        for (i, &(should_be_free, is_free)) in free_indices.iter().enumerate() {
            assert_eq!(
                should_be_free, is_free,
                "Free list missed item, or contains an item it shouldn't: {}",
                i
            );
            if !should_be_free {
                occupied_count += 1;
            }
        }
        assert_eq!(
            self.len, occupied_count,
            "len doesn't reflect the actual number of entries"
        );
    }

    /// Directly query the value of the generation at that index.
    ///
    /// If `index` is greater then `self.capacity()`, then this returns None.
    ///
    /// Advanced usage note: Even generations always indicate an occupied index,
    /// except for 0, which is never a valid generation.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// let mut m = HandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.raw_generation_for_index(h.index()), Some(h.generation()));
    /// ```
    ///
    /// # Caveat
    /// This is a low level feature intended for advanced usage, typically you
    /// do not need to call this function, however doing so is harmless.
    #[inline]
    pub fn raw_generation_for_index(&self, index: usize) -> Option<u16> {
        self.entries.get(index).map(|e| e.gen)
    }

    pub(crate) fn raw_remove(&mut self, index: usize) -> Option<T> {
        let mut e = &mut self.entries[index];
        e.gen = e.gen.wrapping_add(1);
        if e.gen == 0 {
            e.gen = 1;
        }
        e.next = self.next;
        self.next = Some(index as u32);
        self.len -= 1;
        let r = e.payload.take();
        debug_assert!(r.is_some());
        #[cfg(test)]
        {
            self.assert_valid();
        }
        r
    }
}

impl<T> core::ops::Index<Handle> for HandleMap<T> {
    type Output = T;
    fn index(&self, h: Handle) -> &T {
        self.get(h).expect("Invalid handle used in index")
    }
}

impl<T> core::ops::IndexMut<Handle> for HandleMap<T> {
    fn index_mut(&mut self, h: Handle) -> &mut T {
        self.get_mut(h).expect("Invalid handle used in index_mut")
    }
}

/// An iterator that moves out of a HandleMap.
#[derive(Debug)]
pub struct IntoIter<T> {
    inner: alloc::vec::IntoIter<Entry<T>>,
}

impl<T> IntoIterator for HandleMap<T> {
    type IntoIter = IntoIter<T>;
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.entries.into_iter(),
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner
            .try_for_each(|e| {
                if let Some(p) = e.payload {
                    Err(p)
                } else {
                    Ok(())
                }
            })
            .err()
    }
    // TODO: Size hint.
}

#[cold]
fn unlikely_hint() {}

#[derive(Debug, Clone)]
struct Entry<T> {
    next: Option<u32>,
    gen: u16,
    payload: Option<T>,
}

/// An untyped reference to some value. Handles are just a fancy u64.
///
/// Internally these store:
///
/// - A 32-bit index field.
/// - The 16-bit 'generation' of that index (this is incremented both when an
///   item is removed from the index, and when another is inserted).
/// - An extra value typically used to store the ID of their map.
///
/// They're a #[repr(transparent)] wrapper around a u64, so if they need to be
/// passed into C code over the FFI, that can be done directly.
///
/// # Advanced Details
///
/// Typical use of this library expects that you just treat these as opaque,
/// however you're free to inspect and construct them as you please (with
/// `from_raw` and `from_raw_parts`), with the caveat that using the API to do
/// so could cause the map to return non-sensical answers.
///
/// That said, should you want to do so, you absolutely can.
///
/// Some important notes if you're going to construct these:
///
/// - Valid indices should always be between 0 and i32::max_value.
///
/// - Generations for occupied indexs have a even value, and for empty indexs
///   have an odd value. The zero generation is always skipped, and is never
///   considered valid.
///
/// - If used with a HandleMap, the `meta` value must match the map they came
///   from.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Handle(u64);

impl Handle {
    /// A constant for the default (null) handle. Never valid or returned by any
    /// map.
    pub const EMPTY: Handle = Handle::from_raw(0);

    /// Returns the index value of this handle.
    ///
    /// While a usize is returned, this value is guaranteed to be 32 bits.
    ///
    /// # Caveat
    ///
    /// This is a low level feature intended for advanced usage, typically you
    /// do not need to access this value, however doing so is harmless.
    #[inline]
    pub const fn index(self) -> usize {
        (self.0 as u32) as usize
    }

    /// Returns the generation value of this handle.
    ///
    /// # Caveat
    ///
    /// This is a low level feature intended for advanced usage, typically you
    /// do not need to access this value, however doing so is harmless.
    #[inline]
    pub const fn generation(self) -> u16 {
        (self.0 >> 48) as u16
    }

    /// Returns the metadata field of this handle. This is an alias for
    /// `map_id`, as in the common case, this is what the metadata field is used
    /// for.
    ///
    /// See [`Handle::meta`] for more info.
    #[inline]
    pub const fn map_id(self) -> u16 {
        (self.0 >> 32) as u16
    }

    /// Returns the metadata field of this handle.
    ///
    /// If used with a [`HandleMap`] (instead of directly coming from a
    /// [`HandleAlloc`]), this is the `id` of the `HandleMap` which constructed
    /// this handle. If used with a HandleAlloc, then the value has no meaning
    /// aside from whatever you assign to it -- it's 16 free bits you can use
    /// for whatever tagging you want.
    ///
    /// # Caveat
    ///
    /// This is a low level feature intended for advanced usage, typically you
    /// do not need to access this value, however doing so is harmless.
    #[inline]
    pub const fn meta(self) -> u16 {
        (self.0 >> 32) as u16
    }

    /// Construct a handle from the separate parts.
    ///
    /// # Warning
    /// This is a feature intended for advanced usage. An attempt is made to
    /// cope with dubious handles, but it's almost certainly possible to pierce
    /// the abstraction veil of the HandleMap if you use this.
    ///
    /// However, it should not be possible to cause memory unsafety -- this
    /// crate has no unsafe code.
    #[inline]
    #[allow(clippy::cast_lossless)] // const fn
    pub const fn from_raw_parts(index: usize, generation: u16, meta: u16) -> Self {
        Handle((index as u32 as u64) | ((meta as u64) << 32) | ((generation as u64) << 48))
    }

    /// Construct a handle from it's internal `u64` value.
    ///
    /// # Layout
    ///
    /// The 64 bit value is interpreted as such. It's recommended that you
    /// instead use `from_raw_parts` to construct these in cases where this is
    /// relevant, though.
    ///
    /// ```text
    /// [16 bits of generation | 16 bits of map id | 32 bit index]
    /// MSB                                                    LSB
    /// ```
    ///
    /// # Warning
    ///
    /// This is a feature intended for advanced usage. An attempt is made to
    /// cope with dubious handles, but it's almost certainly possible to pierce
    /// the abstraction veil of the HandleMap if you use this.
    ///
    /// However, it should not be possible to cause memory unsafety -- this
    /// crate has no unsafe code.
    #[inline]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Get the internal u64 representation of this handle.
    ///
    /// # Caveat
    ///
    /// This is a low level feature intended for advanced usage, typically you
    /// do not need to access this value, however doing so is harmless.
    ///
    /// # Layout
    ///
    /// The layout of the returned value is as such:
    ///
    /// ```text
    /// [16 bits of generation | 16 bits of map id | 32 bit index]
    /// MSB                                                    LSB
    /// ```
    #[inline]
    pub const fn into_raw(self) -> u64 {
        self.0
    }

    /// Get the internal parts of this handle.
    ///
    /// Equivalent to `(self.index(), self.generation(), self.meta())`
    ///
    /// # Caveat
    ///
    /// This is a low level feature intended for advanced usage, typically you
    /// do not need to access this value, however doing so is harmless.
    #[inline]
    pub fn decompose(self) -> (u32, u16, u16) {
        (self.index() as u32, self.generation(), self.meta())
    }
}

struct EntriesDebug<'a, T>(&'a HandleMap<T>);

impl<'a, T> core::fmt::Debug for EntriesDebug<'a, T>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self.0.iter_with_handles()).finish()
    }
}

impl<T> core::fmt::Debug for HandleMap<T>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HandleMap")
            .field("id", &self.id)
            .field("entries", &EntriesDebug(self))
            .finish()
    }
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("meta", &self.meta())
            .field("generation", &self.generation())
            .field("index", &self.index())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_parts() {
        let h = Handle::from_raw_parts(0, 0, 0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.meta(), 0);
        assert_eq!(h.meta(), h.map_id());

        let h = Handle::from_raw_parts(!0, 0, 0);
        assert_eq!(h.index(), (!0u32) as usize);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.meta(), 0);
        assert_eq!(h.meta(), h.map_id());
        assert_eq!(h.decompose(), (h.index() as u32, h.generation(), h.meta()));

        assert_eq!(Handle::from_raw(h.into_raw()), h);

        let h = Handle::from_raw_parts(0, !0, 0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), !0);
        assert_eq!(h.meta(), 0);
        assert_eq!(h.meta(), h.map_id());

        let h = Handle::from_raw_parts(0, 0, !0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.meta(), !0);
        assert_eq!(h.meta(), h.map_id());

        let h = Handle::from_raw_parts(!0, !0, !0);
        assert_eq!(h.index(), (!0u32) as usize);
        assert_eq!(h.generation(), !0);
        assert_eq!(h.meta(), !0);
        assert_eq!(h.meta(), h.map_id());
    }

    #[derive(PartialEq, Debug, Clone, Copy)]
    pub(crate) struct Foobar(pub(crate) usize);

    #[test]
    fn test_correct_value_single() {
        let mut map = HandleMap::new();
        let handle = map.insert(Foobar(1234));
        assert_eq!(map.get(handle).unwrap(), &Foobar(1234));
        map.remove(handle).unwrap();
        assert_eq!(map.get(handle), None);
    }

    #[test]
    fn test_indexing() {
        let mut map = HandleMap::new();
        let handle = map.insert(Foobar(5454));
        assert_eq!(map[handle].0, 5454);
        map[handle] = Foobar(6767);
        assert_eq!(map[handle].0, 6767);
    }

    #[test]
    fn test_correct_value_multiple() {
        let mut map = HandleMap::new();
        let handle1 = map.insert(Foobar(1234));
        let handle2 = map.insert(Foobar(4321));
        assert_eq!(map.get(handle1).unwrap(), &Foobar(1234));
        assert_eq!(map.get(handle2).unwrap(), &Foobar(4321));
        map.remove(handle1).unwrap();
        assert_eq!(map.get(handle1), None);
        assert_eq!(map.get(handle2).unwrap(), &Foobar(4321));
    }

    #[test]
    fn test_wrong_map() {
        let mut map1 = HandleMap::new();
        let mut map2 = HandleMap::new();

        let handle1 = map1.insert(Foobar(1234));
        let handle2 = map2.insert(Foobar(1234));

        assert_eq!(map1.get(handle1).unwrap(), &Foobar(1234));
        assert_eq!(map2.get_mut(handle2).unwrap(), &mut Foobar(1234));

        assert_eq!(map1.get(handle2), None);
        assert_eq!(map2.get_mut(handle1), None);
        assert_eq!(handle1.meta(), map1.map_id());
        map1.raw_set_map_id(5);
        let h = map1.insert(Foobar(3));
        assert_eq!(h.meta(), map1.map_id());
        assert_eq!(h.meta(), 5);
    }

    #[test]
    fn test_bad_index() {
        let map: HandleMap<Foobar> = HandleMap::new();
        assert_eq!(map.get(Handle::from_raw_parts(100, 2, map.id)), None);
    }

    #[test]
    fn test_wrong_gen() {
        let mut map: HandleMap<usize> = HandleMap::new();
        let h = map.insert(3);
        map.remove(h).unwrap();
        assert_eq!(map.get(h), None);
        assert_eq!(map.remove(h), None);
    }

    #[test]
    fn test_resizing() {
        let mut map = HandleMap::new();
        let mut handles = alloc::vec![];
        // should trigger resize many times
        for i in 0..2000 {
            handles.push(map.insert(Foobar(i)))
        }
        for (i, &h) in handles.iter().enumerate() {
            assert_eq!(map[h], Foobar(i));
            assert_eq!(map.remove(h).unwrap(), Foobar(i));
        }
        let mut handles2 = alloc::vec![];
        for i in 2050..4100 {
            // Not really related to this test, but it's convenient to check this here.
            let h = map.insert(Foobar(i));
            handles2.push(h);
        }
        for (i, (&h0, h1)) in handles.iter().zip(handles2).enumerate() {
            // It's still a stale version, even though the index is occupied again.
            assert_eq!(map.get(h0), None);
            assert_eq!(map.get(h1).unwrap(), &Foobar(i + 2050));
        }
    }
    #[test]
    fn test_reserve() {
        let mut map = HandleMap::with_capacity(10);
        map.reserve(30);
        let mut handles = alloc::vec![];
        for i in 0..10 {
            handles.push(map.insert(Foobar(i)));
            map.reserve(3);
        }
        map.reserve(0);
        for i in 0..10 {
            handles.push(map.insert(Foobar(i + 10)))
        }
        map.reserve(map.capacity());
        for (i, &h) in handles.iter().enumerate() {
            assert_eq!(map[h], Foobar(i));
            assert_eq!(map.remove(h).unwrap(), Foobar(i));
        }
        let mut handles2 = alloc::vec![];
        for i in 20..30 {
            let h = map.insert(Foobar(i));
            handles2.push(h);
        }
        map.reserve(50);
        for (i, (&h0, h1)) in handles.iter().zip(handles2).enumerate() {
            assert_eq!(map.get(h0), None);
            assert_eq!(map.get(h1).unwrap(), &Foobar(i + 20));
        }
    }

    #[test]
    fn test_clear() {
        let mut map = HandleMap::new();
        map.clear(); // no-op.
        for _ in 0..2 {
            let mut handles = alloc::vec![];
            for i in 0..120 {
                handles.push(map.insert(Foobar(i)))
            }
            map.clear();
            for h in handles.iter() {
                assert_eq!(map.get(*h), None);
            }
        }
    }

    #[test]
    fn test_iters() {
        use alloc::collections::BTreeMap;
        let (mut map, handles) = mixed_handlemap();

        assert_eq!(map.len(), handles.len());
        let handle_to_foo: BTreeMap<Handle, usize> = handles.iter().copied().collect();
        let foo_to_handle: BTreeMap<usize, Handle> =
            handles.iter().copied().map(|t| (t.1, t.0)).collect();

        assert_eq!(handle_to_foo.len(), handles.len());
        assert_eq!(foo_to_handle.len(), handles.len());

        // iter
        let mut count = 0;
        for i in map.iter() {
            count += 1;
            assert!(foo_to_handle.contains_key(&i.0));
        }
        assert_eq!(count, handles.len());

        // iter_mut
        let mut count = 0;
        for i in map.iter_mut() {
            count += 1;
            assert!(foo_to_handle.contains_key(&i.0));
        }
        assert_eq!(count, handles.len());

        // into_iter
        let mut count = 0;
        for i in map.clone() {
            count += 1;
            assert!(foo_to_handle.contains_key(&i.0));
        }
        assert_eq!(count, handles.len());

        // iter_with_handles
        let mut count = 0;
        for (h, i) in map.iter_with_handles() {
            count += 1;
            assert!(foo_to_handle.contains_key(&i.0));
            assert_eq!(handle_to_foo[&h], i.0);
        }
        assert_eq!(count, handles.len());

        // iter_mut_with_handles
        let mut count = 0;
        for (h, i) in map.iter_mut_with_handles() {
            count += 1;
            assert!(foo_to_handle.contains_key(&i.0));
            assert_eq!(handle_to_foo[&h], i.0);
        }
        assert_eq!(count, handles.len());
    }

    pub(crate) fn mixed_handlemap() -> (HandleMap<Foobar>, Vec<(Handle, usize)>) {
        let mut handles = alloc::vec![];
        let mut map = HandleMap::with_capacity(10);
        let mut c = 0;
        for &sp in &[2, 3, 5] {
            for _ in 0..100 {
                c += 1;
                handles.push((map.insert(Foobar(c)), c));
                assert_eq!(map.len(), handles.len());
            }
            let mut i = 0;
            while i < handles.len() {
                map.remove(handles.swap_remove(i).0).unwrap();
                assert_eq!(map.len(), handles.len());
                i += sp;
            }
        }
        (map, handles)
    }
    #[test]
    fn test_find() {
        let mut m = HandleMap::new();
        let mut v = alloc::vec![];
        for i in 0..10usize {
            v.push(m.insert(i));
        }
        for (i, h) in v.iter().enumerate() {
            assert_eq!(m.find_handle(&i), Some(*h));
            assert!(m.contains_key(*h));
        }
        m.clear();
        assert!(m.is_empty());
        for (i, h) in v.iter().enumerate() {
            assert_eq!(m.find_handle(&i), None);
            assert!(!m.contains_key(*h));
        }
    }
    #[test]
    fn test_dbg() {
        let mut m = HandleMap::new_with_map_id(0);
        m.insert(0u32);
        assert_eq!(
            alloc::format!("{:?}", m),
            "HandleMap { id: 0, entries: {Handle { meta: 0, generation: 2, index: 0 }: 0} }"
        );
    }
}
