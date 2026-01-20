# $env:DDOS_SERVER_TEST_CSV="C:\Users\lanh2\Documents\HocTap\NamHoc_2023_2024\luanvan\flower\ddos-attack\data\test_final.csv"
$env:DDOS_LABEL_MAP="C:\Users\lanh2\Documents\HocTap\NamHoc_2023_2024\luanvan\flower\ddos-attack\data\label_map.json"

flower-superlink --insecure --fleet-api-address 0.0.0.0:9092 --control-api-address 0.0.0.0:9093
