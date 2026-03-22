-- Supabase schema bootstrap for AgriVision
-- Run this in Supabase SQL Editor (or as a migration).

create extension if not exists pgcrypto;

create table if not exists public.user_info (
    user_id uuid primary key default gen_random_uuid(),
    username text not null unique,
    password text not null,
    privilege text not null default 'USER',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_user_info_username on public.user_info(username);

create table if not exists public.uploaded_images (
    image_id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.user_info(user_id) on delete cascade,
    filename text not null,
    original_path text not null,
    heatzone_path text,
    status text not null default 'processed' check (status in ('uploaded', 'processing', 'processed', 'completed', 'failed')),
    uploaded_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_uploaded_images_user_id on public.uploaded_images(user_id);
create index if not exists idx_uploaded_images_uploaded_at on public.uploaded_images(uploaded_at desc);

create table if not exists public.analysis_results (
    result_id uuid primary key default gen_random_uuid(),
    image_id uuid not null unique references public.uploaded_images(image_id) on delete cascade,
    analysis_json jsonb not null default '{}'::jsonb,
    summary text,
    health_band text,
    health_score double precision,
    confidence double precision,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
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

drop trigger if exists trg_user_info_updated_at on public.user_info;
create trigger trg_user_info_updated_at
before update on public.user_info
for each row execute function public.set_updated_at();

drop trigger if exists trg_uploaded_images_updated_at on public.uploaded_images;
create trigger trg_uploaded_images_updated_at
before update on public.uploaded_images
for each row execute function public.set_updated_at();

drop trigger if exists trg_analysis_results_updated_at on public.analysis_results;
create trigger trg_analysis_results_updated_at
before update on public.analysis_results
for each row execute function public.set_updated_at();
