-- Supabase schema bootstrap for AgriVision
-- Run this in Supabase SQL Editor (or as a migration).

create table if not exists public.images (
    image_id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.user_info(user_id) on delete cascade,
    original_path text not null,
    ai_path text,
    status text not null default 'uploaded' check (status in ('uploaded', 'processing', 'completed', 'failed')),
    uploaded_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_images_user_id on public.images(user_id);
create index if not exists idx_images_uploaded_at on public.images(uploaded_at desc);

create table if not exists public.analysis_results (
    result_id uuid primary key default gen_random_uuid(),
    image_id uuid not null references public.images(image_id) on delete cascade,
    index_type text not null,
    summary text,
    confidence numeric(5,2),
    stats jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_results_image_id on public.analysis_results(image_id);
create index if not exists idx_results_created_at on public.analysis_results(created_at desc);

create table if not exists public.recommendations (
    recommendation_id uuid primary key default gen_random_uuid(),
    result_id uuid not null references public.analysis_results(result_id) on delete cascade,
    title text not null,
    details text not null,
    priority text not null default 'medium' check (priority in ('low', 'medium', 'high')),
    created_at timestamptz not null default now()
);

create index if not exists idx_recommendations_result_id on public.recommendations(result_id);

create table if not exists public.activity_logs (
    log_id uuid primary key default gen_random_uuid(),
    user_id uuid references public.user_info(user_id) on delete set null,
    action text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_activity_logs_user_id on public.activity_logs(user_id);
create index if not exists idx_activity_logs_created_at on public.activity_logs(created_at desc);

-- Optional trigger helper for updated_at
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_images_updated_at on public.images;
create trigger trg_images_updated_at
before update on public.images
for each row execute function public.set_updated_at();