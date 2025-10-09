# [SAS Program for CDC Growth Charts](https://www.cdc.gov/growth-chart-training/hcp/computer-programs/sas.html)

## Includes CDC's Extended BMI-for-Age Growth Charts

### What to know

This SAS program calculates percentiles and z-scores for body mass index (BMI), height, weight, and other metrics for children and adolescents 2 to 20 years. Results are based on CDC's 2000 Growth Charts for those without obesity and CDC's 2022 Extended BMI-for-Age Growth Charts for those with obesity.

### Overview

This SAS program calculates percentiles and z-scores (standard deviations [SD]) for a child's sex and age for BMI, weight, height, and head circumference from CDC growth charts. Weight-for-height z-scores and BMI percentiles are also calculated. The program allows for the identification of outliers, but keep in mind that outliers might be correct values. Consider reviewing outliers, also referred to as "extreme values", so you can decide whether to include or exclude them.

This SAS program calculates BMI z-scores and percentiles for children and adolescents with and without obesity (BMI at or above the 95th percentile). In 2022, BMI z-scores and percentiles calculated for those with obesity changed to use extended BMI z-scores and percentiles. The z-scores and percentiles calculated for children with obesity from this SAS program will differ from those calculated from earlier versions of this SAS program (before 2022).

For infants and children younger than 2 years, the World Health Organization (WHO) growth charts are recommended. Programs on the WHO and CDC websites are based on these WHO growth charts and should be used instead of this SAS program for children under 2.

#### Non-SAS Users

If you are using R, you can use these R programs to calculate z-scores and percentiles. If you're not using SAS or R, you can download CDCref_d.csv and cdc-source-code.sas.

### Instructions for SAS users

**Step 1**: Download the SAS program (`cdc-source-code.sas`) and the reference data file (`CDCref_d.sas7bdat`). Move these files to a folder (directory) that SAS can access. For the following example, the files are in:

```
c:\sas_growth_charts
```

Example SAS code corresponding to Steps 2 to 6 below. After downloading the SAS code and the reference data, cut and paste the following 4 lines into your SAS program. You will likely need to change the libname and %include statements to point at the folder (directory) for the downloaded files. You'll also probably need to rename and recode variables, as explained in Steps 2 to 6.

```sas
libname refdir 'c:\sas_growth_charts';
data mydata; set whatever-your-original-dataset-is-named;
%include 'c:\sas_growth_charts\cdc-source-code.sas';
proc means data=_cdcdata; run;
```

**Step 2**: Create a libname statement in your SAS program to point at the folder location of `CDCref_d.sas7bdat`. An example would be:

```sas
libname refdir 'c:\sas_growth_charts';
```

Note: the SAS code expects this name to be refdir. Make sure you specify this in the libname statement.

**Step 3**: Set your existing data that contains height, weight, sex, age, and other variables into a temporary dataset named `mydata`. Use the information in Table 1 to rename and code the variables.

#### Table 1. Guidance for SAS users on renaming and coding variables in dataset

| Variable  | Description of variables and coding in the input dataset, mydata |
|-----------|-----------------------------------------------------------------|
| agemos    | Months of age. Agemos must be in your dataset, and the program assumes that you know the number of months to the nearest day. For example, if a child were born on Oct 1, 2007, and examined on Nov 15, 2011, the child’s age would be 1506 days or 49.48 (1506 / 30.4375) months. In everyday usage, this child’s age would be 4 years or 49 months. However, if 49 months were used for all children between 49.0 and < 50 months of age, then most of the calculated z-scores would be too high because, on average, these children would be taller and heavier than children who are 49.0 months of age. If only the completed number of months is known (as in NHANES), add 0.5 to the age so that the maximum error would be 15 days. If age represents the completed years (e.g., 13 years), multiply by 12 and add 6. If age is in days, divide by 30.4375. |
| sex       | Sex must be coded as 1 for boys and 2 for girls.               |
| height    | Height (cm). Height is either standing height (for children ≥ 24 months of age) or recumbent length (< 24 months). If standing height was measured for children under 24 months of age, you should add 0.8 cm to these values (see page 8 of 2000 CDC Growth Charts for the United States: Methods and Development). If recumbent length was measured for children ≥ 24 months, subtract 0.8 cm. |
| weight    | Weight (kg)                                                    |
| bmi       | BMI [Weight (kg) / Height (m)²]. The program calculates BMI if it is not present in your data but will not overwrite BMI if present. |
| headcir   | Head circumference (cm)                                        |

Z-scores and percentiles for the anthropometric variables not in mydata (or that are missing) will be coded as missing (.) in the output dataset, `_cdcdata`. It is unlikely that the SAS code will overwrite variables in your dataset, but you should avoid having variable names that begin with an underscore or with 'mod_'.

**Step 4**: Copy and paste the following line into your SAS program after the line (or lines) in Step 3.

```sas
%include 'c:\sas_growth_charts\cdc-source-code.sas'; run;
```

If necessary, change this statement to point at the folder or directory containing the downloaded `cdc-source-code.sas` file. The %include will run your data through `cdc-source-code.sas` and create a dataset named `_cdcdata`.

**Step 5**: The output dataset, `_cdcdata`, contains your original data and z-scores, percentiles, and flags for extreme values shown in Table 2. Additional information on the extreme z-scores is given in the Extreme Values, Implausible Values, and Data Errors section.

**Step 6**: Examine the new dataset, `_cdcdata`, to verify that the z-scores and other variables have been created. If z-scores and percentiles for a variable in your dataset are unexpectedly missing, make sure your dataset is named `mydata` and your variables are named and coded as shown in Table 1. The program will not modify your original data but adds new variables to your dataset. Table 2 shows the names and descriptions of several variables in `_cdcdata`.

#### Table 2. Z-scores, percentiles, and extreme (possibly implausible, BIV) values in the output dataset, `_cdcdata`

| Description | Percentile | Z-score | Modified Z-score to Identify Extreme Values | Flag for Extreme Values | Low (Flag = -1) | High (Flag = +1) |
|-------------|------------|---------|--------------------------------------------|--------------------------|-------------|-----------------------------------------|
| Weight-for-age for children aged from 0 to < 240 months | wapct | waz | mod_waz | _bivwt | < -5 | >8 |
| Height-for-age for children aged from 0 to < 240 months | hapct | haz | mod_haz | _bivht | < -5 | >4 |
| Weight-for-height for children with heights from 45 to 121 cm (these heights approximately correspond to ages 0 to 6 years) | whpct | whz | mod_whz | _bivwh | < -4 | >8 |
| BMI-for-age for children aged 24 to < 240 months | bmipct | bmiz | mod_bmiz | _bivbmi | < -4 | >8 |
| Head circumference-for-age for children aged from 0 to < 36 months | headcpct | headcz | mod_headcz | _bivhc | < -5 | >5 |
| Original calculations for BMI | original_bmipct | original_bmiz | | | | |

**Notes**:
- Several cut points were changed in 2016.
- The names of the modified z-scores were changed in Dec 2022. Previously, they began with '_F.'
- See sections on LMS Method and Extended BMI percentiles and z-scores.

#### Table 3. Additional variables in the output dataset, `_cdcdata`

| Variable | Description |
|----------|-------------|
| bmi50 and bmi95 | Sex- and age-specific 50th and 95th percentiles of BMI in the CDC growth charts |
| bmip50 and bmip95 | BMI expressed as a percentage of CDC’s 50th and 95th percentiles |
| bmi120 | The BMI value that is 120% of the CDC 95th percentile |

### LMS method

The LMS (lambda, mu, sigma) method calculates BMI z-scores as:

```
Z-score = ((BMI / M)^L - 1) / (L × S) [equation 1]
```

The L (transformation for normality), M (median), and S (coefficient of variation) values for the CDC growth charts, which vary by sex and month of age, are in `CDCref_d.sas7bdat`. These z-scores are then transformed into percentiles with the SAS `probnorm` function. For example, a z-score of 1.645 is the 95th percentile. For more information on the LMS method, developed by Tim Cole and P.J Greene in the 1990s, see:

- [LMS method references]

Also see sex- and age-specific L, M, and S values.

The LMS method for BMI results in a curvilinear relation between BMI and BMIz, as shown in the following figure for children aged 5, 12, and 18 years. The range of BMIs (x-axis) corresponds to those observed in NHANES 1999–2000 through 2017–2018. At low BMIs, a small change in BMI results in a sizeable BMIz change. In contrast, at very high BMIs, the same BMI change results in a much smaller BMIz change. This is most evident among males aged 12 years and females aged 18 years.

![Relation of BMI to BMIz at three ages. BMIz was calculated using the LMS values and equation 1.](https://www.cdc.gov/growth-chart-training/media/images/Figure1-bmiz-3-ages.svg)

Further, if a child's BMI is very large relative to the median BMI, then (BMI ÷ M)^L in the LMS equation approaches 0, and the maximum possible BMIz value at that sex/age is (-1) ÷ (L × S). For most ages older than 5 years, the maximum possible BMIz, regardless of the BMI magnitude, is less than 4.0 SDs. Further, among males aged 7 to 15 years and females aged 15 to 19 years, BMIz cannot be greater than 3.3 SDs, limiting the usefulness of these z-scores in characterizing the extremely high BMIs (for example, 40 kg/m² or higher).

The 2000 CDC Growth Charts are based on data collected from 1963 to 1980 for most children and adolescents. It was advised that extrapolation beyond the 97th percentile be done cautiously. Further, the 2000 CDC Growth Charts' BMI z-scores were not intended for use among children and adolescents with extremely high BMI values. Several studies have highlighted the limitations of LMS-calculated BMIz in characterizing very high BMIs.

### Extended BMI percentiles and z-scores

To explore alternative metrics for BMI, the National Center for Health Statistics convened a workshop in 2018 and published a 2022 report that evaluated several alternatives to LMS-BMIz. This report recommended using "extended BMIz" and "extended BMI percentiles" to characterize the BMIs of children and adolescents with obesity (BMI ≥ 95th percentile for a child's sex and age).

These extended metrics were constructed from the BMIs of children and adolescents with obesity in the CDC growth chart reference population and more recent NHANES surveys (through 2015–2016). These BMI data were modeled within each sex and 6-month age group as a half-normal distribution, a truncated normal distribution with only values at or to the right of the peak having a probability density greater than 0. Characterizing these distributions' shape parameter, sigma, allows calculating BMI percentiles for children and adolescents with obesity, even those with extremely high BMIs. These percentiles can then be transformed into z-scores.

Updates from December 2022 make it easier to use these extended metrics. The calculated values for BMIz and BMI percentile (`bmiz` and `bmipct`) in the SAS program combine the LMS-based values for children and adolescents without obesity with the extended values for children and adolescents with obesity. As a result, the original BMI metrics—constructed using only the L, M, and S parameters—have been renamed as `original_bmiz` and `original_bmipct`. Note that `bmiz` and `original_bmiz` (and `bmipct` and `original_bmipct`) are identical for children and adolescents without obesity.

The following figure shows the relation of BMI to the original and new (extended) values of BMIz. The dashed lines represent the original, LMS-based BMI z-scores from the 2000 CDC growth charts. The solid lines represent the extended `bmiz` values for BMIs greater than or equal to the 95th percentile (z-score = 1.645 SDs).

![Relation of BMI to Original and New (Extended) BMIz at three ages. Dashed lines represent the original z-scores; solid lines are the new z-scores](https://www.cdc.gov/growth-chart-training/media/images/Figure2-bmiz-ext-bmiz-ages.svg)

Among children and adolescents without obesity, the LMS-based z-scores and the new BMI z-score are identical. At higher BMIs, the relation of BMI to `bmiz` is fairly linear and does not approach a horizontal asymptote. However, the extended BMIz values are lower than the original values for some BMIs above the 95th percentile, which is most evident for males aged 5 and 18 years in the figure. These lower values arise because children and adolescents with obesity in more recent NHANES surveys have higher BMIs than those in the original CDC reference population.

Although the original LMS-based BMIz was problematic for z-scores greater than 1.88 (97th percentile), extended BMIz has a maximum value of slightly less than 8.21 because of the limitations of computer precision. Percentiles very close to 100, such as 99.999999999999993, are indistinguishable from 100 and cannot be converted to a z-score. SAS outputs these z-scores as missing, but the SAS macro converts these z-scores to 8.21. An extended BMIz of 8.20 corresponds to BMIs of about 33 at age 3 years, 55 at age 12 years, and 82 at age 17 years. Actual BMIs this high are very uncommon. These children and adolescents could be monitored using their BMI value.

### Severe obesity

Because the original LMS-based z-scores for very high BMIs resulted in percentiles that differ from those estimated from the data, a BMI ≥ 120% of the CDC 95th percentile has been widely used for classifying severe obesity since 2013. This cut-point is approximately equal to the empirical 99th percentile in the growth charts. However, among older adolescents, a BMI can be ≥ 35 kg/m² but be less than 120% of the 95th percentile. Therefore, severe obesity is defined as either a BMI ≥ 120% of the 95th percentile or a BMI ≥ 35 kg/m². This aligns with guidelines from the American Heart Association and the American Academy of Pediatrics.

The SAS program outputs the variable, `bmip95`, which expresses a child's BMI as a percentage of the CDC 95th percentile for the child's age and sex, which can range from below 50 to more than 220. For example, a `bmip95` of 140 would indicate that a child has a BMI equal to 1.4 times the 95th percentile. You can also calculate the arithmetic difference between a child's BMI and the CDC 95th percentile. For example, the CDC 95th percentile for a 60-month-old boy is 17.9 kg/m². If this boy had a BMI of 21.3 kg/m², the arithmetic difference would be 3.4 kg/m² (21.3 – 17.9), and `bmip95` would be 119% (100 × 21.3/17.9).

### Extreme values, implausible values, and data errors

As explained in the modified z-scores documentation, the SAS code also calculates modified z-scores that you can use to identify extreme values that may be errors. These modified z-scores were computed by extrapolating one-half of the distance between 0 and +2 (or between 0 and -2) z-scores to the distribution's tails. Although these z-scores were developed to identify outliers at a single examination, they have been incorporated into algorithms for cleaning longitudinal data.

The output from the SAS program contains biologically implausible value (BIV) flag variables for weight, height, and BMI that are coded as -1 (modified z-score is very low), +1 (modified z-score is very high), or 0 (modified z-score is between these two cut-points). These BIV flags in the output dataset, `_cdcdata`, were included in Table 2. It is essential to realize that an extreme value is not necessarily incorrect. The value should be further examined, possibly along with other characteristics of the child.

The upper thresholds for the modified z-score cut-points were initially based on a 1995 WHO publication but were changed in 2016. Several papers showed that these cut-points excluded many children whose weight, height, or BMI were very likely to have been recorded correctly. These BIVs can flag potentially problematic data points, but the BIV cut-points are not a gold standard. The cut-points were chosen to balance including extreme values likely to be correct and excluding those likely to be incorrect.

These paper results led to an increase in the upper cut-points in 2016 from:

- +5 to +8 for modified z-scores for weight and BMI.
- +3 to +4 for modified z-scores for height.

These new z-score cut-points roughly correspond to the modified z-scores for the maximum values of the body size measures among children aged 2 to 18 years in NHANES at many, but not all, ages. However, please be careful in using these cut-points to exclude data, as different decisions could alter the prevalence of obesity and severe obesity by up to 1%.

Other cut-points for the modified z-scores may be more appropriate based on additional information in your data. For example, does a child with an extremely high BMI also have a high skinfold thickness or arm circumference or is very tall? If so, the very high BMI value is more likely to be correct. Similarly, in a longitudinal study or an analysis of electronic health records (EHRs), one could assess whether a child has extreme values of weight and BMI at multiple examinations.

Although +8 SDs is the threshold for a high BMI BIV, two young (less than aged 5 years) boys in NHANES (2005–2006 and 2017–2018) have a modified BMIz greater than 11 SDs. Further, EHR datasets that comprise millions of children indicate that many children consistently have a modified BMIz between 10 and 12 SDs at consecutive examinations. Growthcleanr, an R package, helps identify errors in longitudinal datasets containing multiple records for each child.

You can use the modified z-scores to construct other cut-points for extreme values rather than relying on the BIV flag variables. For example, if you feel using a BMI-for-age cut-point of +8 SDs would exclude many values likely to be correct, then you could use `mod_bmiz > 10` as the definition of a high BMI BIV. This could be recoded as:

```sas
if -5 <= mod_bmiz <= 10 then _bivbmi=0; *plausible;
else if mod_bmiz > 10 then _bivbmi=1; *high BIV;
else if . < mod_bmiz < -5 then _bivbmi= -1; *low BIV;
```

### References
1. Kuczmarski RJ, Ogden CL, Guo SS, Grummer-Strawn LM, Flegal KM, Mei Z, Wei R, Curtin LR, Roche AF, Johnson CL. 2000 CDC Growth Charts for the United States: methods and development. Vital Health Stat 11 2002;11:1–190.
1. Flegal KM, Cole TJ. Construction of LMS parameters for the Centers for Disease Control and Prevention 2000 Growth Charts. Natl Health Stat Rep 2013;9:1–3.
1. Flegal KM, Wei R, Ogden CL, Freedman DS, Johnson CL, Curtin LR. Characterizing extreme values of body mass index-for-age by using the 2000 Centers for Disease Control and Prevention Growth Charts. Am J Clin Nutr 2009;90:1314–20.
1. Woo JG. Using body mass index Z-score among severely obese adolescents: a cautionary note. Int J Pediatr Obes 2009;4:405–10.
1. Freedman DS, Butte NF, Taveras EM, Lundeen EA, Blanck HM, Goodman AB, Ogden CL. BMI z-Scores are a poor indicator of adiposity among 2- to 19-year-olds with very high BMIs, NHANES 1999-2000 to 2013-2014. Obes Silver Spring 2017;25:739–46.
1. Freedman DS, Berenson GS. Tracking of BMI z Scores for Severe Obesity. Pediatrics. 2017;140:e20171072.
1. Hales C, Freedman DS, Akinbami L, Wei R, Ogden CL. Using CDC growth charts to assess and monitor weight status in children and adolescents with extremely high BMI. Natl Cent Health Stat Vital Health Stat.2 2022;197.
1. Wei R, Ogden CL, Parsons VL, Freedman DS, Hales CM. A method for calculating BMI z-scores and percentiles above the 95th percentile of the CDC growth charts. Ann Hum Biol Taylor & Francis. 2020;47:514–21.
1. Kelly AS, Barlow SE, Rao G et. al, American Heart Association Atherosclerosis, Hypertension, and Obesity in the Young Committee of the Council on Cardiovascular Disease in the Young, Council on Nutrition, Physical Activity and Metabolism, and Council on Clinical Cardiology. Severe obesity in children and adolescents: identification, associated health risks, and treatment approaches: a scientific statement from the American Heart Association. Circulation. 2013;128:1689–712.
1. Armstrong SC, Bolling CF, Michalsky MP, Reichard KW, Haemer MA, Muth ND, Rausch JC, Rogers VW, Heiss KF, Besner GE, et al. Pediatric Metabolic and Bariatric Surgery: Evidence, Barriers, and Best Practices. Pediatrics. 2019;144:e20193223.
1. Daymont C, Ross ME, Russell Localio A, Fiks AG, Wasserman RC, Grundmeier RW. Automated identification of implausible values in growth data from pediatric electronic health records. J Am Med Inform Assoc. 2017;24:1080–7.
1. World Health Organization. Physical status: the use and interpretation of anthropometry. Report of a WHO Expert Committee. World Health Organ Tech Rep Ser. 1995;854:1–452.
1. Lawman HG, Ogden CL, Hassink S, Mallya G, Vander Veur S, Foster GD. Comparing methods for identifying biologically implausible values in height, weight, and body bass index among youth. Am J Epidemiol. 2015;182:359–65.
1. Freedman DS, Lawman HG, Skinner AC, McGuire LC, Allison DB, Ogden CL. Validity of the WHO cutoffs for biologically implausible values of weight, height, and BMI in children and adolescents in NHANES from 1999 through 2012. Am J Clin Nutr. 2015;102:1000–6.
1. Freedman DS, Lawman HG, Pan L, Skinner AC, Allison DB, McGuire LC, Blanck HM. The prevalence and validity of high, biologically implausible values of weight, height, and BMI among 8.8 million children. Obes Silver Spring. 2016;24:1132–9.