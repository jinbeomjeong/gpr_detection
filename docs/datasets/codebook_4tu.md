# Codebook metadata.csv (4TU GPR 데이터셋 원문 추출)

- 원본: `C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015\00_원본\Codebook.pdf`
- 2쪽, 가로 A4, Microsoft Word for Microsoft 365에서 내보낸 텍스트 PDF. 작성일 2023-08-22
- 추출: pdf-mcp `pdf_read_pages` (pdfplumber 표, `columns_reliable=true`). 1쪽 표와 2쪽 표는 하나의 표가 쪽에서 나뉜 것이다. `Land type`의 값 목록이 1쪽 끝에서 2쪽 첫 줄로 이어지므로 아래에서 한 행으로 합쳤다.
- 원문 그대로 옮겼다(영문). 번역, 해석, 보충은 하지 않았다. `Category` 열은 원문에서 병합 셀이라 각 범주의 첫 행에만 값이 있다.

## 원문 표

| Category | Feature | Description | Values |
|---|---|---|---|
| Construction | Utility surveying objective | The objective set by the construction organizations for the utility surveying activity. | Verify utility maps; Locate specific utilities/appurtenances; Map free subsoil space. |
| | Construction works | The type of construction work the utility surveying is carried out for. | Replacement/remediation/installation of utilities; Excavation; None. |
| | Exact location accuracy required | The level of accuracy of utility locations required by the project organizations. | Yes; No. |
| | Complementary works | Any complementary construction work conducted by the construction organizations. | Controlled directional drilling; Soil preparation / remediation; None. |
| Geophysical | Land use | The socio-economic description of the area in which surveying takes place. | High-density residential; Commercial and industrial; Residential; Non-build up. |
| | Ground condition | The type of subsurface soil. | Sandy; Clayey; Silty; Loamy. |
| | Ground relative permittivity | The relative permittivity of the subsurface soil. | Integer value. |
| | Relative groundwater level | The relative level of the groundwater in relation to the utilities. | Within utility range; Below utility range; Partially within utility range. |
| | Land cover | The type of surface cover the surveying activity is carried out on. | Asphalt; Concrete tiling/bricks; Concrete slabs; Grass/vegetation; Rubble; Sand. |
| | Land type | The functional type of surface the surveying activity is carried out on. | Street/cycling road; Sidewalk; Pedestrian/parking area; Greenery; Construction site. |
| | Terrain levelling | The levelling of the surface. | Flat; Steep. |
| | Terrain smoothness | The smoothness of the surface. | Smooth; Rough. |
| | Weather condition | The weather conditions. | Dry; Rainy. |
| | Presence of anomalies | The presence of rubble, tree roots, polluted soil, or blast-furnace slags in the subsoil. | Yes; No. |
| Utility infrastructure | Amount of utilities | The number of utilities found. | Integer value. |
| | Utility crossing | Whether there are crossing utilities present. | Yes; No. |
| | Utility path linear | Whether the orientation of the utility path is linear. If not, it means the path is curved. | Yes; No. |
| | Utility discipline | The disciplines of the utilities found. | Electricity; OilGasChemicals; Sewer; Water; Telecommunications. |
| | Additional utility information | Additional information about the utilities. | Shielded with a cover; Part of a bundle; Installed in a conduit; Material / diameter transition. |
| | Utility material | The material type of the utilities found. | Longlist of material values. |
| | Utility diameter | The diameter of the utilities found. | Integer value in mm. |

## 원문에 없는 것 (추출 범위 확인용)

Codebook에는 위 표 하나만 있다. 다음 항목은 이 문서에 정의가 없다.

- `LocationID` 열
- ground-truth 그림의 `Distance in meters` 와 `Depth in meters`의 기준점(지표인지, 관 상단·중심인지)
- 측선(Path)과 시굴 트렌치의 위치 관계
- `Ground relative permittivity` 값의 산출 방법과 출처
