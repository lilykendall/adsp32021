# Data Dictionary

The data was aggregated from a number of sources including the American Community Survey
(census.gov), clinicaltrials.gov, and cancer.gov. Most of the data preparation process can
be viewed here.

**Source key**

| Key | Meaning |
| --- | --- |
| (a) | Years 2010–2016 |
| (b) | 2013 Census Estimates |

---

## Target

| Field | Description | Source |
| --- | --- | --- |
| `TARGET_deathRate` | **Dependent variable.** Mean per capita (100,000) cancer mortalities | (a) |

## Cancer incidence & research

| Field | Description | Source |
| --- | --- | --- |
| `avgAnnCount` | Mean number of reported cases of cancer diagnosed annually | (a) |
| `avgDeathsPerYear` | Mean number of reported mortalities due to cancer | (a) |
| `incidenceRate` | Mean per capita (100,000) cancer diagnoses | (a) |
| `studyPerCap` | Per capita number of cancer-related clinical trials per county | (a) |

## Geography & population

| Field | Description | Source |
| --- | --- | --- |
| `Geography` | County name | (b) |
| `popEst2015` | Population of county | (b) |
| `AvgHouseholdSize` | Mean household size of county | (b) |

## Income & poverty

| Field | Description | Source |
| --- | --- | --- |
| `medianIncome` | Median income per county | (b) |
| `binnedInc` | Median income per capita binned by decile | (b) |
| `povertyPercent` | Percent of populace in poverty | (b) |

## Age

| Field | Description | Source |
| --- | --- | --- |
| `MedianAge` | Median age of county residents | (b) |
| `MedianAgeMale` | Median age of male county residents | (b) |
| `MedianAgeFemale` | Median age of female county residents | (b) |

## Education

| Field | Description | Source |
| --- | --- | --- |
| `PctNoHS18_24` | Percent of county residents ages 18–24, highest education attained: less than high school | (b) |
| `PctHS18_24` | Percent of county residents ages 18–24, highest education attained: high school diploma | (b) |
| `PctSomeCol18_24` | Percent of county residents ages 18–24, highest education attained: some college | (b) |
| `PctBachDeg18_24` | Percent of county residents ages 18–24, highest education attained: bachelor's degree | (b) |
| `PctHS25_Over` | Percent of county residents ages 25 and over, highest education attained: high school diploma | (b) |
| `PctBachDeg25_Over` | Percent of county residents ages 25 and over, highest education attained: bachelor's degree | (b) |

## Employment

| Field | Description | Source |
| --- | --- | --- |
| `PctEmployed16_Over` | Percent of county residents ages 16 and over employed | (b) |
| `PctUnemployed16_Over` | Percent of county residents ages 16 and over unemployed | (b) |

## Health coverage

| Field | Description | Source |
| --- | --- | --- |
| `PctPrivateCoverage` | Percent of county residents with private health coverage | (b) |
| `PctPrivateCoverageAlone` | Percent of county residents with private health coverage alone (no public assistance) | (b) |
| `PctEmpPrivCoverage` | Percent of county residents with employee-provided private health coverage | (b) |
| `PctPublicCoverage` | Percent of county residents with government-provided health coverage | (b) |
| `PctPubliceCoverageAlone` | Percent of county residents with government-provided health coverage alone | (b) |

## Race

| Field | Description | Source |
| --- | --- | --- |
| `PctWhite` | Percent of county residents who identify as White | (b) |
| `PctBlack` | Percent of county residents who identify as Black | (b) |
| `PctAsian` | Percent of county residents who identify as Asian | (b) |
| `PctOtherRace` | Percent of county residents who identify in a category which is not White, Black, or Asian | (b) |

## Household & birth

| Field | Description | Source |
| --- | --- | --- |
| `PercentMarried` | Percent of county residents who are married | (b) |
| `PctMarriedHouseholds` | Percent of married households | (b) |
| `BirthRate` | Number of live births relative to number of women in county | (b) |
