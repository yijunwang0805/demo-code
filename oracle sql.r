SELECT h.shopcode,
       TO_CHAR(TRUNC(h.completedate), 'YYYY-MM-DD') SALESDATE,
       av.name SALESCATEGORY,
       CASE
         WHEN av.name LIKE '%diamond%' THEN dsr.name
         WHEN av.name LIKE '%gold%' THEN dwr.name
         ELSE dpr.name END SUBCATEGORY, 
       SUM(bpp.price4) RETAIL, 
       SUM(h.qty) QTY,
       MIN(hd.datetype) DAYTYPE
FROM ruser.V_RETAILORDER h -- sales, date, quantity, take out special sales, special order, and orders that have already paid
JOIN business_pricesinprod_price bpp
  on bpp.singleproductcode = h.singleproductcode
JOIN business_shop_extend spe -- open, close
  ON h.shopcode = spe.shopcode
  AND spe.end_day IS NULL -- take out closed stores
JOIN business_btgoods b
  ON SUBSTR(h.singleproductcode,1,11) = b.code
 AND b.ctrlmode = '0' 
JOIN dict_holiday2 hd
  ON TRUNC(h.completedate) = hd.datetime
JOIN business_attrvalue av -- subcatogory
  ON b.ATTR6 = av.id
  AND av.name NOT IN ('14Kgoldendiamond','14Kgoldenrings','salesmaterial','repair','silver','else') -- take out unneeded subcatory
JOIN v_shop v
  ON h.shopcode = v.code
  AND v.area_name IS NOT NULL AND v.NAME NOT LIKE '%marketingsales%'
  AND v.area_name NOT LIKE '%marketingstores%' AND v.area_name NOT LIKE '%cancel%' AND v.area_name NOT LIKE '%jointmanagement%' -- take out irrelevant stores
JOIN business_btsinglprodu bt --weight
  on bt.code = h.singleproductcode
  AND bt.ATTR6 != 110094 -- non diamond customer service
LEFT JOIN dict_stone_range dsr -- diamond range
  ON bt.attr195 >= dsr.min
 AND bt.attr195 <  dsr.max
 AND dsr.id > 1
LEFT JOIN dict_price_range dpr -- price range
  ON h.pricetag >= dpr.min
 AND h.pricetag <  dpr.max
LEFT JOIN dict_weight_range dwr -- weight range
  ON bt.attr200 >= dwr.min
 AND bt.attr200 <  dwr.max
 AND dpr.id > 1
WHERE h.qty > 0 -- take out return
  AND h.completedate >= DATE'2017-01-01'
  AND av.name NOT LIKE '%PT%'
  AND h.price < 100000
  AND h.completedate >= spe.start_day 
  AND h.completedate < trunc(DATE'", date, "','mm') 
  AND spe.start_day < trunc(DATE'", date, "','mm') - 30 
GROUP BY h.shopcode,
      TRUNC(h.completedate),
      av.name,
      CASE
         WHEN av.name LIKE '%diamond%' THEN dsr.name
         WHEN av.name LIKE '%gold%' THEN dwr.name
         ELSE dpr.name END
ORDER BY h.shopcode,
      TRUNC(h.completedate),
      av.name,
      CASE
         WHEN av.name LIKE '%diamond%' THEN dsr.name
         WHEN av.name LIKE '%gold%' THEN dwr.name
         ELSE dpr.name END
