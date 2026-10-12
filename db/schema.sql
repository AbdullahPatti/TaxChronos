create extension if not exists vector;

create table if not exists chunks (
  id text primary key,
  source text not null,
  title text not null,
  section text not null,
  page int not null,
  content text not null,
  amendments text[] not null default '{}',
  effective_from date not null,
  effective_to date,
  embedding vector(768) not null,
  fts tsvector generated always as (to_tsvector('english', section || ' ' || content)) stored
);

create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx on chunks using gin (fts);
create index if not exists chunks_source_idx on chunks (source);

create table if not exists answers (
  id bigint generated always as identity primary key,
  question text not null,
  language text not null,
  as_of date,
  embedding vector(768) not null,
  response jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists answers_embedding_idx on answers using hnsw (embedding vector_cosine_ops);

alter table chunks enable row level security;
alter table answers enable row level security;

create or replace function hybrid_search(
  query_text text,
  query_embedding vector(768),
  as_of date,
  match_count int default 12,
  rrf_k int default 60
)
returns table (
  id text,
  source text,
  title text,
  section text,
  page int,
  content text,
  amendments text[],
  effective_from date,
  effective_to date,
  score float
)
language sql stable
as $$
  with live as (
    select * from chunks
    where chunks.effective_from <= as_of
      and (chunks.effective_to is null or chunks.effective_to > as_of)
  ),
  dense as (
    select live.id, row_number() over (order by live.embedding <=> query_embedding) as rank
    from live
    order by live.embedding <=> query_embedding
    limit match_count * 4
  ),
  sparse as (
    select live.id, row_number() over (order by ts_rank_cd(live.fts, q) desc) as rank
    from live, websearch_to_tsquery('english', regexp_replace(query_text, '\s+', ' or ', 'g')) q
    where live.fts @@ q
    order by ts_rank_cd(live.fts, q) desc
    limit match_count * 4
  ),
  fused as (
    select coalesce(dense.id, sparse.id) as id,
           coalesce(1.0 / (rrf_k + dense.rank), 0) + coalesce(1.0 / (rrf_k + sparse.rank), 0) as score
    from dense full outer join sparse on dense.id = sparse.id
  )
  select c.id, c.source, c.title, c.section, c.page, c.content, c.amendments,
         c.effective_from, c.effective_to, fused.score
  from fused join chunks c on c.id = fused.id
  order by fused.score desc
  limit match_count
$$;
