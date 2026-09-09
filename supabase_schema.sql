create table if not exists public.quarterly_scores (
    quarter text not null,
    ticker text not null,
    company text not null,
    score double precision not null check (score >= 0 and score <= 100),
    coverage integer not null check (coverage >= 0 and coverage <= 9),
    fetched_at text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (quarter, ticker)
);

create index if not exists quarterly_scores_quarter_idx
    on public.quarterly_scores (quarter);

alter table public.quarterly_scores enable row level security;

-- Recommended for this MVP: keep the table private and use a server-side key
-- stored only in Streamlit Secrets. Do NOT expose the key in browser code.
-- If you later add end-user authentication, replace this with user-scoped RLS policies.
