-- Sprint 7.4: Add sire_name to horses for pedigree features
ALTER TABLE horsebet.horses ADD COLUMN IF NOT EXISTS sire_name TEXT;
CREATE INDEX IF NOT EXISTS idx_horses_sire_name ON horsebet.horses(sire_name);
