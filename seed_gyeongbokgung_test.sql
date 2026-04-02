/* =========================================================
   경복궁 테스트 Seed 데이터
   =========================================================
   [답안지 - 화장실 추천 테스트]

   키오스크          | 위치              | 추천 화장실      | 거리   | 근거
   ─────────────────┼──────────────────┼─────────────────┼───────┼──────────────
   kiosk-south-01  | 광화문 앞          | 화장실_남쪽      | ~67m  | 같은 zone
   kiosk-center-01 | 근정전 옆          | 화장실_중앙      | ~49m  | 같은 zone
   kiosk-north-01  | 향원정 앞          | 화장실_북쪽      | ~23m  | 같은 zone
   kiosk-east-01   | 동쪽 담장 옆       | 화장실_외부      | ~177m | fallback (zone 내 화장실 없음)

   [Zone 구성]
   남쪽구역(SOUTH)  : 광화문~흥례문 구역  (화장실 있음)
   중앙구역(CENTER) : 근정전~경회루 구역  (화장실 있음)
   북쪽구역(NORTH)  : 향원정~신무문 구역  (화장실 있음)
   동쪽구역(EAST)   : 동쪽 담장 구역      (화장실 없음 → fallback 테스트용)

   [좌표 기준]
   - 좌표계: WGS84 (SRID 4326), ST_MakePoint(경도, 위도)
   ========================================================= */

-- 기존 테스트 데이터 초기화 (재실행 시 중복 방지)
DELETE FROM tb_device WHERE device_id LIKE 'kiosk-%';
DELETE FROM tb_place  WHERE site_id IN (SELECT site_id FROM tb_site WHERE name = '경복궁');
DELETE FROM tb_zone   WHERE site_id IN (SELECT site_id FROM tb_site WHERE name = '경복궁');
DELETE FROM tb_site   WHERE name = '경복궁';

-- ─────────────────────────────────────────────────────────
-- 1. Site
-- ─────────────────────────────────────────────────────────
INSERT INTO tb_site (name) VALUES ('경복궁');

-- ─────────────────────────────────────────────────────────
-- 2. Zone (INNER 4개)
--    Polygon 좌표 순서: (경도 위도) - PostGIS WKT 표준
-- ─────────────────────────────────────────────────────────
INSERT INTO tb_zone (site_id, name, code, zone_type, parent_zone_id, area_geometry)
SELECT s.site_id, '남쪽구역', 'SOUTH', 'INNER', NULL,
  ST_GeomFromText('POLYGON((126.974 37.574, 126.980 37.574, 126.980 37.577, 126.974 37.577, 126.974 37.574))', 4326)
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_zone (site_id, name, code, zone_type, parent_zone_id, area_geometry)
SELECT s.site_id, '중앙구역', 'CENTER', 'INNER', NULL,
  ST_GeomFromText('POLYGON((126.972 37.577, 126.980 37.577, 126.980 37.580, 126.972 37.580, 126.972 37.577))', 4326)
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_zone (site_id, name, code, zone_type, parent_zone_id, area_geometry)
SELECT s.site_id, '북쪽구역', 'NORTH', 'INNER', NULL,
  ST_GeomFromText('POLYGON((126.970 37.5795, 126.979 37.5795, 126.979 37.583, 126.970 37.583, 126.970 37.5795))', 4326)
FROM tb_site s WHERE s.name = '경복궁';

-- 동쪽구역: 화장실 없는 zone → fallback 테스트용
INSERT INTO tb_zone (site_id, name, code, zone_type, parent_zone_id, area_geometry)
SELECT s.site_id, '동쪽구역', 'EAST', 'INNER', NULL,
  ST_GeomFromText('POLYGON((126.981 37.576, 126.984 37.576, 126.984 37.580, 126.981 37.580, 126.981 37.576))', 4326)
FROM tb_site s WHERE s.name = '경복궁';

-- ─────────────────────────────────────────────────────────
-- 3. Place - 화장실 (category: TOILET)
-- ─────────────────────────────────────────────────────────
-- zone 내부 화장실 3개
INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '화장실_남쪽', 'TOILET',
  ST_SetSRID(ST_MakePoint(126.9769, 37.5752), 4326)::geography,
  '광화문 매표소 동쪽 화장실. 남녀 구분. 장애인 화장실 포함.'
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '화장실_중앙', 'TOILET',
  ST_SetSRID(ST_MakePoint(126.9765, 37.5775), 4326)::geography,
  '근정전 서쪽 화장실. 남녀 구분. 기저귀 교환대 있음.'
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '화장실_북쪽', 'TOILET',
  ST_SetSRID(ST_MakePoint(126.9750, 37.5800), 4326)::geography,
  '향원정 남동쪽 화장실. 남녀 구분.'
FROM tb_site s WHERE s.name = '경복궁';

-- zone 외부 화장실 (zone_id = NULL) → fallback 테스트용
-- 동쪽 담장 바깥 (lng 126.9800, zone EAST는 126.981부터 시작)
INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '화장실_외부', 'TOILET',
  ST_SetSRID(ST_MakePoint(126.9800, 37.5770), 4326)::geography,
  '경복궁 동쪽 담장 외부 공중화장실. 어느 zone에도 속하지 않음.'
FROM tb_site s WHERE s.name = '경복궁';

-- ─────────────────────────────────────────────────────────
-- 4. Place - 랜드마크 (category: LANDMARK)
-- ─────────────────────────────────────────────────────────
INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '광화문', 'LANDMARK',
  ST_SetSRID(ST_MakePoint(126.9769, 37.5757), 4326)::geography,
  '경복궁의 남쪽 정문. 조선시대 궁궐 문 중 가장 크고 웅장한 문.'
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '근정전', 'LANDMARK',
  ST_SetSRID(ST_MakePoint(126.9762, 37.5775), 4326)::geography,
  '경복궁의 중심 건물. 국왕이 신하들에게 조회를 받던 정전.'
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '경회루', 'LANDMARK',
  ST_SetSRID(ST_MakePoint(126.9745, 37.5781), 4326)::geography,
  '연못 위에 세워진 누각. 외국 사신 접대 및 연회 장소.'
FROM tb_site s WHERE s.name = '경복궁';

INSERT INTO tb_place (site_id, name, category, location, description)
SELECT s.site_id, '향원정', 'LANDMARK',
  ST_SetSRID(ST_MakePoint(126.9751, 37.5801), 4326)::geography,
  '작은 연못 향원지 위의 정자. 경복궁 북쪽의 대표 명소.'
FROM tb_site s WHERE s.name = '경복궁';

-- ─────────────────────────────────────────────────────────
-- 5. Place zone_id 자동 배정 (polygon 안에 속하는 것만)
--    화장실_외부는 어느 zone에도 속하지 않아 zone_id = NULL 유지
-- ─────────────────────────────────────────────────────────
UPDATE tb_place p
SET zone_id     = z.zone_id,
    zone_source = 'MANUAL'
FROM tb_zone z
WHERE p.site_id = z.site_id
  AND p.site_id = (SELECT site_id FROM tb_site WHERE name = '경복궁')
  AND ST_Within(p.location::geometry, z.area_geometry);

-- ─────────────────────────────────────────────────────────
-- 6. Device - 키오스크 4개
-- ─────────────────────────────────────────────────────────
-- kiosk-south-01: 광화문 앞 → 남쪽구역 → 화장실_남쪽 ~67m (같은 zone)
INSERT INTO tb_device (device_id, auth_token_hash, site_id, zone_id, zone_source, location_name, location)
SELECT 'kiosk-south-01',
  'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2',
  s.site_id, z.zone_id, 'MANUAL', '광화문 앞 키오스크',
  ST_SetSRID(ST_MakePoint(126.9770, 37.5758), 4326)::geography
FROM tb_site s JOIN tb_zone z ON z.site_id = s.site_id AND z.code = 'SOUTH'
WHERE s.name = '경복궁';

-- kiosk-center-01: 근정전 옆 → 중앙구역 → 화장실_중앙 ~49m (같은 zone)
INSERT INTO tb_device (device_id, auth_token_hash, site_id, zone_id, zone_source, location_name, location)
SELECT 'kiosk-center-01',
  'b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3',
  s.site_id, z.zone_id, 'MANUAL', '근정전 옆 키오스크',
  ST_SetSRID(ST_MakePoint(126.9760, 37.5773), 4326)::geography
FROM tb_site s JOIN tb_zone z ON z.site_id = s.site_id AND z.code = 'CENTER'
WHERE s.name = '경복궁';

-- kiosk-north-01: 향원정 앞 → 북쪽구역 → 화장실_북쪽 ~23m (같은 zone)
INSERT INTO tb_device (device_id, auth_token_hash, site_id, zone_id, zone_source, location_name, location)
SELECT 'kiosk-north-01',
  'c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4',
  s.site_id, z.zone_id, 'MANUAL', '향원정 앞 키오스크',
  ST_SetSRID(ST_MakePoint(126.9751, 37.5798), 4326)::geography
FROM tb_site s JOIN tb_zone z ON z.site_id = s.site_id AND z.code = 'NORTH'
WHERE s.name = '경복궁';

-- kiosk-east-01: 동쪽 담장 옆 → 동쪽구역 (화장실 없음) → 화장실_외부 ~177m (fallback)
INSERT INTO tb_device (device_id, auth_token_hash, site_id, zone_id, zone_source, location_name, location)
SELECT 'kiosk-east-01',
  'd4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5',
  s.site_id, z.zone_id, 'MANUAL', '동쪽 담장 옆 키오스크',
  ST_SetSRID(ST_MakePoint(126.9820, 37.5772), 4326)::geography
FROM tb_site s JOIN tb_zone z ON z.site_id = s.site_id AND z.code = 'EAST'
WHERE s.name = '경복궁';

-- ─────────────────────────────────────────────────────────
-- 검증 쿼리
-- ─────────────────────────────────────────────────────────

-- [검증1] Place zone 배정 확인 (화장실_외부는 zone명이 비어 있어야 함)
SELECT p.name, p.category, COALESCE(z.name, '(zone 없음)') AS zone명, p.zone_source
FROM tb_place p
LEFT JOIN tb_zone z ON z.zone_id = p.zone_id
WHERE p.site_id = (SELECT site_id FROM tb_site WHERE name = '경복궁')
ORDER BY p.category, p.name;

-- [검증2] 핵심: zone 우선 → 거리 fallback 화장실 추천 (답안지와 비교)
SELECT
  d.device_id                                                     AS 키오스크,
  d.location_name                                                 AS 키오스크_위치,
  COALESCE(z_d.name, '(zone 없음)')                              AS 키오스크_zone,
  p.name                                                          AS 추천_화장실,
  COALESCE(z_p.name, '(zone 없음)')                              AS 화장실_zone,
  ROUND(ST_Distance(d.location, p.location)::numeric, 1)         AS 거리_m,
  CASE WHEN p.zone_id = d.zone_id THEN '같은 zone' ELSE 'fallback' END AS 추천_근거
FROM tb_device d
LEFT JOIN tb_zone z_d ON z_d.zone_id = d.zone_id
CROSS JOIN LATERAL (
  SELECT p2.name, p2.location, p2.zone_id
  FROM tb_place p2
  WHERE p2.site_id = d.site_id
    AND p2.category = 'TOILET'
    AND p2.is_active = TRUE
  ORDER BY
    CASE WHEN p2.zone_id = d.zone_id THEN 0 ELSE 999999 END
    + ST_Distance(d.location, p2.location)
  LIMIT 1
) p
LEFT JOIN tb_zone z_p ON z_p.zone_id = p.zone_id
WHERE d.device_id LIKE 'kiosk-%'
ORDER BY d.device_id;