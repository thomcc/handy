//! This module provides [`TypedHandleMap`][typed::TypedHandleMap] and
//! [`TypedHandle`][typed::TypedHandle]. These are wrappers around of
//! [`HandleMap`] and [`Handle`] which statically prevent trying to use a
//! `Handle` returned from a `HandleMap<T>` to get a value out of a
//! `HandleMap<U>`, instead of allowing it to fail at runtime, as it will with
//! `HandleMap`.
//!
//! For most use cases, this is probably not worth the extra trouble, but it's
//! provided for completeness, and because the definition of `TypedHandle` has
//! some subtle gotchas.
//!
//! These abstractions are thin. Methods exist to go bidirectionally to and from
//! both `TypedHandleMap<T>` to `HandleMap<T>` and `TypedHandle<T>` to `Handle`.
//! You shouldn't need to do this, but restricting it seems needless.

use crate::{Handle, HandleMap};
use core::marker::PhantomData;

/// A `TypedHandleMap` is a wrapper around a [`HandleMap`] which gives you some
/// additional type safety, should you desire.
///
/// It accepts and returns [`TypedHandle`]s, and you can only pass a
/// `TypedHandle<T>` to a `TypedHandleMap<T>` -- attempting to pass it to a
/// `TypedHandleMap<U>` will be statically detected.
///
/// Beyond this, it still can detect use of a handle that came from another map.
///
/// You use it with `TypedHandle`s, which only will accept handles of the
/// correct type. This could be useful if you have several handle maps in your
/// program, and find
///
/// `TypedHandle<T>` is Copy + Send + Sync (and several others) regardless of
/// `T`, which is not true for a naïve implementation of this, so it's provided
/// even though I don't think it's that helpful for most usage (handle maps
/// already detect this at runtime).
#[derive(Clone, Debug)]
pub struct TypedHandleMap<T>(HandleMap<T>);

impl<T> TypedHandleMap<T> {
    /// Create a new typed handle map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// // No allocation is performed by default.
    /// assert_eq!(m.capacity(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self(HandleMap::new())
    }

    /// Create a typed handle map from one which accepts untyped handles.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// # use handy::typed::{TypedHandleMap, TypedHandle};
    /// let mut m: HandleMap<u32> = HandleMap::new();
    /// let h = m.insert(10u32);
    /// let tm = TypedHandleMap::from_untyped(m);
    /// assert_eq!(tm[TypedHandle::from_handle(h)], 10u32);
    /// ```
    #[inline]
    pub fn from_untyped(h: HandleMap<T>) -> Self {
        Self(h)
    }

    /// Convert this map into it's wrapped [`HandleMap`]. See also
    /// [`TypedHandleMap::as_untyped_map`] and [`TypedHandleMap::as_mut_untyped_map`].
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// # use handy::typed::TypedHandleMap;
    /// let mut tm: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let th = tm.insert(10u32);
    /// let m = tm.into_untyped();
    /// assert_eq!(m[th.handle()], 10);
    /// ```
    #[inline]
    pub fn into_untyped(self) -> HandleMap<T> {
        self.0
    }

    /// Create a new typed handle map with the specified capacity.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let m: TypedHandleMap<u32> = TypedHandleMap::with_capacity(10);
    /// // Note that we don't guarantee the capacity will be exact.
    /// // (though in practice it will so long as the requested
    /// // capacity is >= 8)
    /// assert!(m.capacity() >= 10);
    /// ```
    #[inline]
    pub fn with_capacity(c: usize) -> Self {
        Self::from_untyped(HandleMap::with_capacity(c))
    }

    /// Get the number of entries we can hold before reallocation.
    ///
    /// This just calls [`HandleMap::capacity`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// m.insert(10);
    /// assert!(m.capacity() >= 1);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Get the number of occupied entries.
    ///
    /// This just calls [`HandleMap::len`] on our wrapped map.
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// assert_eq!(m.len(), 0);
    /// m.insert(10u32);
    /// assert_eq!(m.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if our length is zero.
    ///
    /// This just calls [`HandleMap::is_empty`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// assert!(m.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get a reference to this map as a [`HandleMap`].
    ///
    /// This shouldn't be necessary except in advanced use, but may allow access
    /// to APIs that aren't mirrored, like most of the `raw` APIs.
    ///
    /// ## Example
    /// ```
    /// # use handy::HandleMap;
    /// # use handy::typed::TypedHandleMap;
    /// let mut tm: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let th = tm.insert(10u32);
    /// assert_eq!(tm.as_untyped_map()[th.handle()], 10);
    /// ```
    #[inline]
    pub fn as_untyped_map(&self) -> &HandleMap<T> {
        &self.0
    }

    /// Get a mutable reference to this map as a [`HandleMap`].
    ///
    /// This shouldn't be necessary except in advanced use, but may allow access
    /// to APIs that aren't mirrored, like most of the `raw` APIs.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut tm = TypedHandleMap::new();
    /// let th = tm.insert(10u32);
    /// tm.as_mut_untyped_map()[th.handle()] = 5;
    /// assert_eq!(tm[th], 5);
    /// ```
    #[inline]
    pub fn as_mut_untyped_map(&mut self) -> &mut HandleMap<T> {
        &mut self.0
    }

    /// Add a new item, returning a handle to it.
    ///
    /// This just calls [`HandleMap::insert`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m = TypedHandleMap::new();
    /// assert_eq!(m.len(), 0);
    /// m.insert(10u32);
    /// assert_eq!(m.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> TypedHandle<T> {
        TypedHandle::from_handle(self.0.insert(value))
    }

    /// Remove the value referenced by this handle from the map, returning it.
    ///
    /// If the handle doesn't point to an entry in the map we return None. This
    /// will happen if:
    ///
    /// - The handle comes from a different map.
    /// - The item it referenced has been removed already.
    /// - It appears corrupt in some other way.
    ///
    /// This just calls [`HandleMap::remove`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// // Present:
    /// assert_eq!(m.remove(h), Some(10));
    /// // Not present:
    /// assert_eq!(m.remove(h), None);
    /// ```
    #[inline]
    pub fn remove(&mut self, handle: TypedHandle<T>) -> Option<T> {
        self.0.remove(handle.h)
    }

    /// Remove all entries in this handle map.
    ///
    /// This just calls [`HandleMap::clear`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// m.clear();
    /// assert_eq!(m.len(), 0);
    /// assert_eq!(m.get(h), None);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Try and get a reference to the item backed by the handle.
    ///
    /// If the handle doesn't point to an entry in the map we return None. This
    /// will happen if:
    ///
    /// - The handle comes from a different map.
    /// - The item it referenced has been removed already.
    /// - It appears corrupt in some other way.
    ///
    /// This just calls [`HandleMap::get`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.get(h), Some(&10));
    /// m.remove(h);
    /// assert_eq!(m.get(h), None);
    /// ```
    #[inline]
    pub fn get(&self, handle: TypedHandle<T>) -> Option<&T> {
        self.0.get(handle.h)
    }

    /// Try and get mutable a reference to the item backed by the handle.
    ///
    /// If the handle doesn't point to an entry in the map we return None. This
    /// will happen if:
    ///
    /// - The handle comes from a different map.
    /// - The item it referenced has been removed already.
    /// - It appears corrupt in some other way.
    ///
    /// This just calls [`HandleMap::get_mut`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// *m.get_mut(h).unwrap() += 1;
    /// assert_eq!(m[h], 11);
    /// // Note: The following is equivalent if you're going to `unwrap` the result of get_mut:
    /// m[h] += 1;
    /// assert_eq!(m[h], 12);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, handle: TypedHandle<T>) -> Option<&mut T> {
        self.0.get_mut(handle.h)
    }

    /// Returns true if the handle refers to an item present in this map.
    ///
    /// This just calls [`HandleMap::contains`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// assert!(m.contains(h));
    /// m.remove(h);
    /// assert!(!m.contains(h));
    /// ```
    #[inline]
    pub fn contains(&self, h: TypedHandle<T>) -> bool {
        self.0.contains_key(h.h)
    }

    /// Returns true if the handle refers to an item present in this map.
    ///
    /// This just calls [`HandleMap::contains_key`] on our wrapped map.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// assert!(m.contains_key(h));
    /// m.remove(h);
    /// assert!(!m.contains_key(h));
    /// ```
    #[inline]
    pub fn contains_key(&self, h: TypedHandle<T>) -> bool {
        self.0.contains_key(h.h)
    }

    /// Search the map for `item`, and if it's found, return a handle to it.
    ///
    /// If more than one value compare as equal to `item`, it's not specified
    /// which we will return.
    ///
    /// Note that this is a naive O(n) search, so if you want this often, you
    /// might want to store the handle as a field on the value.
    ///
    /// This just calls [`HandleMap::find_handle`] on our wrapped map and wraps
    /// the resulting handle.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.find_handle(&10), Some(h));
    /// assert_eq!(m.find_handle(&11), None);
    /// ```
    #[inline]
    pub fn find_handle(&self, item: &T) -> Option<TypedHandle<T>>
    where
        T: PartialEq,
    {
        self.0.find_handle(item).map(TypedHandle::from_handle)
    }

    /// Reserve space for `sz` additional items.
    ///
    /// This just calls [`HandleMap::reserve`] on our wrapped map.
    ///
    /// ## Example
    ///
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// assert_eq!(m.capacity(), 0);
    /// m.reserve(10);
    /// assert!(m.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, sz: usize) {
        self.0.reserve(sz)
    }

    /// Get an iterator over every occupied slot of this map.
    ///
    /// See also `iter_with_handles` if you want the handles during
    /// iteration.
    ///
    /// This just calls [`HandleMap::iter`] on our wrapped map.
    ///
    /// ## Example
    ///
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// m.insert(10u32);
    /// assert_eq!(*m.iter().next().unwrap(), 10);
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a {
        self.0.iter()
    }

    /// Get a mut iterator over every occupied slot of this map.
    ///
    /// See also `iter_mut_with_handles` if you want the handles during
    /// iteration.
    ///
    /// This just calls [`HandleMap::iter_mut`] on our wrapped map.
    ///
    /// ## Example
    ///
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// for v in m.iter_mut() {
    ///     *v += 1;
    /// }
    /// assert_eq!(m[h], 11);
    /// ```
    #[inline]
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.0.iter_mut()
    }

    /// Get an iterator over every occupied slot of this map, as well as a
    /// handle which can be used to fetch them later.
    ///
    /// This just calls [`HandleMap::iter_with_handles`] on our wrapped map and
    /// wraps the resulting handles.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// # let m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// for (h, v) in m.iter_with_handles() {
    ///     println!("{:?} => {}", h, v);
    /// }
    /// ```
    #[inline]
    pub fn iter_with_handles<'a>(&'a self) -> impl Iterator<Item = (TypedHandle<T>, &'a T)> + 'a {
        self.0
            .iter_with_handles()
            .map(|(h, v)| (TypedHandle::from_handle(h), v))
    }

    /// Get a mut iterator over every occupied slot of this map, as well as a
    /// handle which can be used to fetch them later.
    ///
    /// This just calls [`HandleMap::iter_mut_with_handles`] on our wrapped map and
    /// wraps the resulting handles
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// # let mut m = TypedHandleMap::<u32>::new();
    /// for (h, v) in m.iter_mut_with_handles() {
    ///     *v += 1;
    ///     println!("{:?}", h);
    /// }
    /// ```
    #[inline]
    pub fn iter_mut_with_handles<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (TypedHandle<T>, &'a mut T)> + 'a {
        self.0
            .iter_mut_with_handles()
            .map(|(h, v)| (TypedHandle::from_handle(h), v))
    }

    /// If `index` refers to an occupied entry, return a `Handle` to it.
    /// Otherwise, return None.
    ///
    /// This just calls [`HandleMap::handle_for_index`] on our wrapped map and wraps
    /// the resulting handle.
    ///
    /// ## Example
    /// ```
    /// # use handy::typed::TypedHandleMap;
    /// let mut m: TypedHandleMap<u32> = TypedHandleMap::new();
    /// let h = m.insert(10u32);
    /// assert_eq!(m.handle_for_index(h.index()), Some(h));
    /// ```
    #[inline]
    pub fn handle_for_index(&self, index: usize) -> Option<TypedHandle<T>> {
        self.0.handle_for_index(index).map(TypedHandle::from_handle)
    }
}

/// A `TypedHandle` is a wrapper around a [`Handle`] which gives you some
/// additional type safety, should you desire. You use it with a
/// `TypedHandleMap`, which only will accept handles of the correct type. This
/// could be useful if you have several handle maps in your program, and find
///
/// `TypedHandle<T>` is Copy + Send + Sync (and several others) regardless of
/// `T`, which is not true for a naïve implementation of this, so it's provided
/// even though I don't think it's that helpful for most usage (handle maps
/// already detect this at runtime).
#[repr(transparent)]
pub struct TypedHandle<T> {
    h: Handle,
    _marker: PhantomData<fn() -> T>,
}

impl<T> TypedHandle<T> {
    /// The `TypedHandle` equivalent of [`Handle::EMPTY`].
    pub const EMPTY: Self = Self::from_handle(Handle::EMPTY);

    /// Construct a typed handle from an untyped [`Handle`].
    ///
    /// This typically shouldn't be necessary if you're using typed maps
    /// exclusively, but could be useful  when building abstractions on top of
    /// handles.
    #[inline]
    pub const fn from_handle(h: Handle) -> Self {
        Self {
            h,
            _marker: Self::BOO,
        }
    }

    // Rust (as of the current version) doesn't allow using function pointers
    // (e.g. `fn()`) in const functions. This triggers when constructing a
    // PhantomData, even though it probably shouldn't. To get around this, we
    // just instantiate it outside of the const fn, which surprisingly works.
    const BOO: PhantomData<fn() -> T> = PhantomData;

    /// Access the wrapped untyped handle.
    ///
    /// This typically shouldn't be necessary if you're using typed maps
    /// exclusively, but could be useful when building abstractions on top of
    /// handles, as well as for accessing the accessors on [`Handle`] which are
    /// not otherwise directly exposed on `TypedHandle`.
    #[inline]
    pub const fn handle(self) -> Handle {
        self.h
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
    pub const fn from_raw_parts(index: usize, generation: u16, meta: u16) -> Self {
        Self::from_handle(Handle::from_raw_parts(index, generation, meta))
    }

    /// Construct a handle from it's internal `u64` value.
    ///
    /// See the documentation for [`Handle::from_raw`] for further info.
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
        Self::from_handle(Handle::from_raw(value))
    }

    /// Get the internal u64 representation of this handle.
    ///
    /// See the documentation for [`Handle::into_raw`] for further info.
    #[inline]
    pub const fn into_raw(self) -> u64 {
        self.h.0
    }

    /// Returns the index value of this handle.
    ///
    /// While a usize is returned, this value is guaranteed to be 32 bits.
    ///
    /// See the documentation for [`Handle::index`] for further info.
    #[inline]
    pub const fn index(self) -> usize {
        self.h.index()
    }

    /// Returns the generation value of this handle.
    ///
    /// See the documentation for [`Handle::generation`] for further info.
    #[inline]
    pub const fn generation(self) -> u16 {
        self.h.generation()
    }

    /// Returns the metadata field of this handle.
    ///
    /// See the documentation for [`Handle::meta`] for further info.
    #[inline]
    pub const fn meta(self) -> u16 {
        self.h.meta()
    }

    /// Returns the metadata field of this handle. This is an alias for
    /// `map_id`, as in the common case, this is what the metadata field is used
    /// for.
    ///
    /// See [`Handle::meta`] for more info.
    #[inline]
    pub const fn map_id(self) -> u16 {
        self.h.map_id()
    }
}

impl<T> core::ops::Index<TypedHandle<T>> for TypedHandleMap<T> {
    type Output = T;
    fn index(&self, h: TypedHandle<T>) -> &T {
        self.get(h).expect("Invalid handle used in index")
    }
}

impl<T> core::ops::IndexMut<TypedHandle<T>> for TypedHandleMap<T> {
    fn index_mut(&mut self, h: TypedHandle<T>) -> &mut T {
        self.get_mut(h).expect("Invalid handle used in index_mut")
    }
}

impl<T> Default for TypedHandleMap<T> {
    // #[derive()] only works if T is also Default, so open-code this
    fn default() -> Self { Self::new() }
}

// The automatically derived trait implementations place a bound on T,
// which defeats the whole point of using a handle.
impl<T> Clone for TypedHandle<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            h: self.h,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for TypedHandle<T> {}
impl<T> Eq for TypedHandle<T> {}

impl<T> PartialEq for TypedHandle<T> {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        self.h.0 == o.h.0
    }
    #[inline]
    #[allow(clippy::partialeq_ne_impl)] // derive includes it, so so shall I.
    fn ne(&self, o: &Self) -> bool {
        self.h.0 != o.h.0
    }
}

impl<T> PartialOrd for TypedHandle<T> {
    #[inline]
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        self.h.0.partial_cmp(&o.h.0)
    }
    #[inline]
    fn lt(&self, o: &Self) -> bool {
        self.h.0 < o.h.0
    }
    #[inline]
    fn le(&self, o: &Self) -> bool {
        self.h.0 <= o.h.0
    }
    #[inline]
    fn ge(&self, o: &Self) -> bool {
        self.h.0 >= o.h.0
    }
    #[inline]
    fn gt(&self, o: &Self) -> bool {
        self.h.0 > o.h.0
    }
}

impl<T> IntoIterator for TypedHandleMap<T> {
    type IntoIter = crate::IntoIter<T>;
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> Ord for TypedHandle<T> {
    #[inline]
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        self.h.0.cmp(&o.h.0)
    }
}

impl<T> Default for TypedHandle<T> {
    #[inline]
    fn default() -> Self {
        Self {
            h: Handle::EMPTY,
            _marker: PhantomData,
        }
    }
}

impl<T> core::hash::Hash for TypedHandle<T> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, h: &mut H) {
        core::hash::Hash::hash(&self.h, h)
    }
}

impl<T> core::fmt::Debug for TypedHandle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("TypedHandle").field(&self.h).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_handle_parts() {
        let h = TypedHandle::<()>::from_raw_parts(0, 0, 0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.meta(), 0);
        assert_eq!(h.meta(), h.map_id());

        let h = TypedHandle::<()>::from_raw_parts(!0, 0, 0);
        assert_eq!(h.index(), (!0u32) as usize);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.meta(), 0);
        assert_eq!(h.meta(), h.map_id());

        assert_eq!(TypedHandle::<()>::from_raw(h.into_raw()), h);

        let h = TypedHandle::<()>::from_raw_parts(0, !0, 0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), !0);
        assert_eq!(h.meta(), 0);
        assert_eq!(h.meta(), h.map_id());

        let h = TypedHandle::<()>::from_raw_parts(0, 0, !0);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.meta(), !0);
        assert_eq!(h.meta(), h.map_id());

        let h = TypedHandle::<()>::from_raw_parts(!0, !0, !0);
        assert_eq!(h.index(), (!0u32) as usize);
        assert_eq!(h.generation(), !0);
        assert_eq!(h.meta(), !0);
        assert_eq!(h.meta(), h.map_id());
    }

    use crate::tests::Foobar;

    #[test]
    fn test_correct_value_single() {
        let mut map = TypedHandleMap::new();
        let handle = map.insert(Foobar(1234));
        assert_eq!(map.get(handle).unwrap(), &Foobar(1234));
        map.remove(handle).unwrap();
        assert_eq!(map.get(handle), None);
        let handle = map.as_mut_untyped_map().insert(Foobar(1234));
        assert_eq!(
            map.get(TypedHandle::from_handle(handle)).unwrap(),
            &Foobar(1234)
        );
    }

    #[test]
    fn test_indexing() {
        let mut map = TypedHandleMap::new();
        let handle = map.insert(Foobar(5454));
        assert_eq!(map[handle].0, 5454);
        map[handle] = Foobar(6767);
        assert_eq!(map[handle].0, 6767);
    }

    #[test]
    fn test_correct_value_multiple() {
        let mut map = TypedHandleMap::new();
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
        let mut map1 = TypedHandleMap::new();
        let mut map2 = TypedHandleMap::new();

        let handle1 = map1.insert(Foobar(1234));
        let handle2 = map2.insert(Foobar(1234));

        assert_eq!(map1.get(handle1).unwrap(), &Foobar(1234));
        assert_eq!(map2.get_mut(handle2).unwrap(), &mut Foobar(1234));

        assert_eq!(map1.get(handle2), None);
        assert_eq!(map2.get_mut(handle1), None);
    }

    #[test]
    fn test_bad_index() {
        let map: TypedHandleMap<Foobar> = TypedHandleMap::new();
        assert_eq!(
            map.get(TypedHandle::<Foobar>::from_raw_parts(
                100,
                2,
                map.as_untyped_map().map_id()
            )),
            None
        );
    }

    #[test]
    fn test_reserve() {
        let mut map = TypedHandleMap::<u32>::with_capacity(10);
        let cap0 = map.capacity();
        map.reserve(cap0 + 10);
        assert!(map.capacity() >= cap0 + 10);
    }

    #[test]
    fn test_clear() {
        let mut map = HandleMap::new();
        map.insert(5u32);
        assert!(map.len() == 1);
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn test_iters() {
        use alloc::collections::BTreeMap;
        let (map, handles) = crate::tests::mixed_handlemap();
        let mut map = TypedHandleMap::from_untyped(map);
        let handles = handles
            .into_iter()
            .map(|(h, v)| (TypedHandle::<Foobar>::from_handle(h), v))
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(map.len(), handles.len());
        let handle_to_foo: BTreeMap<TypedHandle<Foobar>, usize> = handles.iter().copied().collect();
        let foo_to_handle: BTreeMap<usize, TypedHandle<Foobar>> =
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

    #[test]
    fn test_find() {
        let mut m = TypedHandleMap::new();
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
    fn test_handle_traits() {
        fn verify<T>()
        where
            T: Clone
                + Copy
                + PartialEq
                + PartialOrd
                + Eq
                + Ord
                + core::hash::Hash
                + Default
                + Send
                + Sync,
        {
        }
        verify::<TypedHandle<u32>>();
        verify::<TypedHandle<*const u32>>();
        verify::<TypedHandle<core::cell::UnsafeCell<u32>>>();
        verify::<TypedHandle<alloc::vec::Vec<u32>>>();
    }

    // Note: fails to compile if we have the variance wrong.
    #[allow(dead_code, unused_assignments, unused_variables)]
    fn check_handle_variance<'a, 'b: 'a>(mut x: TypedHandle<&'a u32>, y: TypedHandle<&'b u32>) {
        // Requires covariance
        x = y;
    }

    #[test]
    // need to actually invoke the clone impl, and I don't feel like splitting this.
    #[allow(clippy::clone_on_copy, clippy::cognitive_complexity)]
    fn test_trait_impls() {
        use core::cmp::Ordering;
        use core::hash::Hash;
        type TH = TypedHandle<()>;
        assert!(TH::from_raw(3) == TH::from_raw(3));
        assert!(TH::from_raw(3) != TH::from_raw(4));

        assert!(!(TH::from_raw(3) != TH::from_raw(3)));
        assert!(!(TH::from_raw(3) == TH::from_raw(4)));

        assert!(TH::from_raw(3) < TH::from_raw(4));
        assert!(TH::from_raw(4) > TH::from_raw(3));

        assert!(!(TH::from_raw(4) < TH::from_raw(4)));
        assert!(!(TH::from_raw(4) < TH::from_raw(3)));

        assert!(!(TH::from_raw(4) > TH::from_raw(4)));
        assert!(!(TH::from_raw(3) > TH::from_raw(4)));

        assert!(TH::from_raw(3) <= TH::from_raw(4));
        assert!(TH::from_raw(3) <= TH::from_raw(3));

        assert!(TH::from_raw(4) >= TH::from_raw(3));
        assert!(TH::from_raw(4) >= TH::from_raw(4));

        assert!(!(TH::from_raw(5) <= TH::from_raw(4)));
        assert!(!(TH::from_raw(4) >= TH::from_raw(5)));

        assert_eq!(
            TH::from_raw(4).partial_cmp(&TH::from_raw(4)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            TH::from_raw(5).partial_cmp(&TH::from_raw(4)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            TH::from_raw(5).partial_cmp(&TH::from_raw(6)),
            Some(Ordering::Less)
        );

        assert_eq!(TH::from_raw(4).cmp(&TH::from_raw(4)), Ordering::Equal);
        assert_eq!(TH::from_raw(5).cmp(&TH::from_raw(4)), Ordering::Greater);
        assert_eq!(TH::from_raw(5).cmp(&TH::from_raw(6)), Ordering::Less);

        assert_eq!(TH::from_raw(5).clone(), TH::from_raw(5));

        let mut h = H_DEFAULT;
        TH::from_raw(10).hash(&mut h);
        let mut h2 = H_DEFAULT;
        Handle::from_raw(10).hash(&mut h2);
        assert_eq!(h.0, h2.0);

        assert_eq!(
            &alloc::format!("{:?}", TH::from_raw_parts(10, 20, 30)),
            "TypedHandle(Handle { meta: 30, generation: 20, index: 10 })",
        );
    }

    // Rather than using the deprecated siphash code, just implement a crappy fnv1
    struct HashTester(u64);
    const H_DEFAULT: HashTester = HashTester(0xcbf2_9ce4_8422_2325);
    impl core::hash::Hasher for HashTester {
        fn write(&mut self, bytes: &[u8]) {
            for b in bytes {
                self.0 = self.0.wrapping_mul(0x100_0000_01b3);
                self.0 ^= u64::from(*b);
            }
        }
        fn finish(&self) -> u64 {
            self.0
        }
    }
}
