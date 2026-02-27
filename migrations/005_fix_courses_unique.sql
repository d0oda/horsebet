-- Fix duplicate courses/jockeys and add UNIQUE constraints for proper upserts
CREATE UNIQUE INDEX IF NOT EXISTS idx_courses_name_jp ON horsebet.courses(name_jp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trainers_name_jp ON horsebet.trainers(name_jp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jockeys_name_jp ON horsebet.jockeys(name_jp);
