pub const MULTIMODAL_CHUNK_SIZE: usize = 512;

pub fn schedule_chunks<T>(token_embeddings: &[T]) -> impl Iterator<Item = &[T]> {
    token_embeddings.chunks(MULTIMODAL_CHUNK_SIZE)
}
