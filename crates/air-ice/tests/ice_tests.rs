use air_ice::paged_cache::PagedKvCache;
use air_ice::kv_anchor::KVAnchor;

#[test]
fn test_paged_cache_allocation() {
    let mut cache = PagedKvCache::new(10);
    let _block = cache.allocate_block(1).unwrap();
    assert_eq!(cache.get_block_table(1).unwrap().len(), 1);
    cache.free_sequence(1);
}

#[test]
fn test_kv_anchor() {
    let mut anchor = KVAnchor::new();
    let initial_hash = anchor.current_hash();
    anchor.update(42, &[1, 2, 3, 4]);
    assert_ne!(anchor.current_hash(), initial_hash);
}
