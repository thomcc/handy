#![no_std]
#![allow(clippy::let_and_return)]
#![deny(unsafe_code)]
//! # `handy`
//!
//! `handy` provides handles and handle maps. This is a fairly useful data
//! structure for rust code, since it can help you work around borrow checker
//! issues.
//!
//! Essentially, `Handle` and `HandleMap` are a more robust version of the
//! pattern where instead of storing a reference to a &T directly, you instead
//! store a `usize` which indicates where it is in some `Vec`. I claim they're
//! more robust because:
//!
//! - They can detect if you try to use a handle in a map other than the one
//!   that provided it.
//!
//! - If you remove an item from the HandleMap, the handle map won't let you use
//!   the stale handle to get whatever value happens to be in that slot at the
//!   time.
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

/// A collection that tells *you* what your value's key should be.
///
/// When inserting a value, it gives you back a [`Handle`] which can be used to
/// fetch that value at later time. Other than that, it's somewhat similar
/// to other collection types.
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
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            len: 0,
            next: None,
            // end: 0,
            id: SOURCE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Create a new handle map with the specified capacity.
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
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of occupied entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if our length is zero
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Add a new item, returning a handle to it.
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
        let res = Handle::from_parts(index as u32, e.gen, self.id);
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
    /// - It appears corrupt in some other way (For example, it's Handle::default())
    pub fn remove(&mut self, handle: Handle) -> Option<T> {
        self.handle_check_mut(handle)?;
        let index = handle.index();
        let mut e = &mut self.entries[index as usize];
        e.gen += 1;
        debug_assert!(e.next.is_none());
        e.next = self.next;
        self.next = Some(index);
        self.len -= 1;
        let r = e.payload.take();
        #[cfg(test)]
        {
            self.assert_valid();
        }
        r
    }

    /// Remove all entries in this handle map.
    pub fn clear(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let update_gen = move |e: &mut Entry<T>| {
            if (e.gen & 1) == 0 {
                e.gen += 1;
            } else {
                e.gen += 2;
            }
            if e.gen == 0 {
                e.gen = 1;
            }
        };
        update_gen(&mut self.entries[0]);
        self.entries[0].payload = None;
        self.entries[0].next = None;
        for i in 1..self.entries.len() {
            update_gen(&mut self.entries[i]);
            self.entries[i].next = Some((i - 1) as u32);
            self.entries[i].payload = None;
        }
        self.next = Some((self.entries.len() - 1) as u32);
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
    ///   Handle::default())
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
    ///   Handle::default())
    #[inline]
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        self.handle_check_mut(handle)
            .and_then(|e| e.payload.as_mut())
    }

    /// Returns true if the handle refers to an item present in this map.
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
    #[inline]
    pub fn find_handle(&self, item: &T) -> Option<Handle>
    where
        T: PartialEq,
    {
        for (i, e) in self.entries.iter().enumerate() {
            if e.payload.as_ref() == Some(item) {
                return Some(Handle::from_parts(i as u32, e.gen, self.id));
            }
        }
        None
    }

    /// Reserve space for `sz` additional items.
    pub fn reserve(&mut self, sz: usize) {
        self.grow(self.len() + sz);
    }

    /// Get an iterator over every occupied slot of this map.
    ///
    /// See also `iter_with_handles` if you want the handles during
    /// iteration.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a {
        self.entries.iter().filter_map(|e| e.payload.as_ref())
    }

    /// Get a mut iterator over every occupied slot of this map.
    ///
    /// See also `iter_mut_with_handles` if you want the handles during
    /// iteration.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.entries.iter_mut().filter_map(|e| e.payload.as_mut())
    }

    /// Get an iterator over every occupied slot of this map, as well as a
    /// handle which can be used to fetch them later.
    pub fn iter_with_handles<'a>(&'a self) -> impl Iterator<Item = (Handle, &'a T)> + 'a {
        self.entries.iter().enumerate().filter_map(move |(i, e)| {
            e.payload
                .as_ref()
                .map(|p| (Handle::from_parts(i as u32, e.gen, self.id), p))
        })
    }

    /// Get a mut iterator over every occupied slot of this map, as well as a
    /// handle which can be used to fetch them later.
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
                    .map(|p| (Handle::from_parts(i as u32, gen, id), p))
            })
    }

    #[inline]
    fn handle_check(&self, handle: Handle) -> Option<&Entry<T>> {
        if handle.source() != self.id {
            unlikely_hint();
            return None;
        }
        let i = handle.index() as usize;
        if i >= self.entries.len() {
            unlikely_hint();
            return None;
        }
        let e = &self.entries[i];
        let gen = handle.gen();
        if e.gen != gen || (gen & 1) != 0 {
            unlikely_hint();
            None
        } else {
            Some(e)
        }
    }

    #[inline]
    fn handle_check_mut(&mut self, handle: Handle) -> Option<&mut Entry<T>> {
        if handle.source() != self.id {
            unlikely_hint();
            return None;
        }
        let i = handle.index() as usize;
        if i >= self.entries.len() {
            unlikely_hint();
            return None;
        }
        let e = &mut self.entries[i];
        let gen = handle.gen();
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
            let n = self.grow_unlikely(self.capacity() + 1);
            debug_assert!(self.next == Some(n as u32));
            n
        }
    }

    #[cold]
    fn grow_unlikely(&mut self, req_cap: usize) -> usize {
        self.grow(req_cap)
    }

    // note: returns `self.next` unwrapped.
    fn grow(&mut self, need: usize) -> usize {
        if need <= self.capacity() {
            return self.next.expect("bug") as usize;
        }
        let cap = (self.capacity() * 2).max(need).max(16);
        assert!(cap <= i32::max_value() as usize, "Capacity overflow");

        self.entries.reserve(cap - self.entries.len());
        // If we're empty, this adds the first entry and free list tail.
        // Otherwise, it lets us use simpler computation in the extend call
        // below. Win/win.
        self.entries.push(Entry {
            next: self.next,
            payload: None,
            gen: 1,
        });

        let current_cap = self.capacity();
        self.entries.extend((current_cap..cap).map(|i| Entry {
            next: Some((i - 1) as u32),
            payload: None,
            gen: 1,
        }));
        let next = (self.entries.len() - 1) as u32;
        self.next = Some(next);
        #[cfg(test)]
        {
            self.assert_valid();
        }
        next as usize
    }

    #[cfg(test)]
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
        assert!(number_of_ends <= 1);

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
/// Internally they store:
///
/// - The index into the map.
/// - The ID of their map.
/// - The 'generation' of that index (this is incremented both when an item is
///   removed from the index, and when another is inserted).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Handle(u64);

impl Handle {
    /// A constant for the null handle. Never valid or returned by any map.
    pub const EMPTY: Handle = Handle(0);

    #[inline]
    pub(crate) const fn index(self) -> u32 {
        self.0 as u32
    }
    #[inline]
    pub(crate) const fn gen(self) -> u16 {
        (self.0 >> 48) as u16
    }

    #[inline]
    pub(crate) const fn source(self) -> u16 {
        (self.0 >> 32) as u16
    }

    #[inline]
    #[allow(clippy::cast_lossless)] // const fn
    pub(crate) const fn from_parts(index: u32, gen: u16, source: u16) -> Self {
        Handle((index as u64) | ((source as u64) << 32) | ((gen as u64) << 48))
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
            .field("map_id", &self.source())
            .field("gen", &self.gen())
            .field("index", &self.index())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_parts() {
        let h = Handle::from_parts(0, 0, 0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.gen(), 0);
        assert_eq!(h.source(), 0);

        let h = Handle::from_parts(!0, 0, 0);
        assert_eq!(h.index(), !0u32);
        assert_eq!(h.gen(), 0);
        assert_eq!(h.source(), 0);

        let h = Handle::from_parts(0, !0, 0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.gen(), !0);
        assert_eq!(h.source(), 0);

        let h = Handle::from_parts(0, 0, !0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.gen(), 0);
        assert_eq!(h.source(), !0);

        let h = Handle::from_parts(!0, !0, !0);
        assert_eq!(h.index(), !0);
        assert_eq!(h.gen(), !0);
        assert_eq!(h.source(), !0);
    }

    #[derive(PartialEq, Debug, Clone, Copy)]
    struct Foobar(usize);

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
        assert_eq!(map2.get(handle2).unwrap(), &Foobar(1234));

        assert_eq!(map1.get(handle2), None);
        assert_eq!(map2.get(handle1), None);
    }

    #[test]
    fn test_bad_index() {
        let map: HandleMap<Foobar> = HandleMap::new();
        assert_eq!(map.get(Handle::from_parts(100, 2, map.id)), None);
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
            // It's still a stale version, even though the slot is occupied again.
            assert_eq!(map.get(h0), None);
            assert_eq!(map.get(h1).unwrap(), &Foobar(i + 2050));
        }
    }

    #[test]
    fn test_clear() {
        let mut map = HandleMap::new();
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

    fn mixed_handlemap() -> (HandleMap<Foobar>, Vec<(Handle, usize)>) {
        let mut handles = alloc::vec![];
        let mut map = HandleMap::new();
        let mut c = 0;
        for &sp in &[2, 3, 5] {
            for _ in 0..100 {
                c += 1;
                handles.push((map.insert(Foobar(c)), c))
            }
            let mut i = 0;
            while i < handles.len() {
                map.remove(handles.swap_remove(i).0).unwrap();
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
        for (i, h) in v.iter().enumerate() {
            assert_eq!(m.find_handle(&i), None);
            assert!(!m.contains_key(*h));
        }
    }

}
