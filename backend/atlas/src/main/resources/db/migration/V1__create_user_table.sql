-- Atlas accounts. Only the Java service writes here; the Python side never
-- reads it, it receives user_id over gRPC from a verified JWT.
--
-- email is stored normalised (trimmed, lower-case) and the check enforces it,
-- so uniqueness cannot be dodged by case. Usernames keep their case for display
-- and are unique case-insensitively.

create table user_table (
    user_id       uuid         primary key,
    username      varchar(32)  not null,
    email         varchar(254) not null,
    password_hash varchar(255) not null,
    created_at    timestamptz  not null default now(),
    updated_at    timestamptz  not null default now(),
    is_active     boolean      not null default true,
    constraint user_table_email_normalised check (email = lower(btrim(email)) and length(email) >= 3),
    constraint user_table_username_format check (username ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$'),
    constraint user_table_password_hash_encoded check (password_hash like '$argon2id$%')
);

create unique index user_table_email_key on user_table (email);
create unique index user_table_username_key on user_table (lower(username));
