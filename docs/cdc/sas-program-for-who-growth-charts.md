# [SAS Program for WHO Growth Charts](https://www.cdc.gov/growth-chart-training/hcp/computer-programs/sas-who.html)

> [!NOTE]
> **WHAT TO KNOW** This SAS program calculates the percentiles and z-scores (standard deviations) for a child’s sex and age from birth to 2 years based on the World Health Organization (WHO) growth charts. The calculations are for body mass index (BMI), weight, height, skinfold thicknesses (triceps and subscapular), arm circumference, head circumference and weight-for-height.

## Overview

This SAS program calculates the z-scores (standard deviations) and percentiles for a child’s sex and age from birth to 2 years. The calculations are for body mass index (BMI), weight, height, skinfold thicknesses (triceps and subscapular), arm circumference, and head circumference based on [WHO growth charts](https://www.who.int/toolkits/child-growth-standards/standards). Weight-for-height z-scores and percentiles are also calculated.

Observations with extreme values ([absolute z-scores above 5 or 6](https://www.who.int/tools/child-growth-standards/software)) are flagged as biologically implausible. Although WHO provides [several macros and a PC program](https://www.who.int/toolkits/child-growth-standards/software) for these calculations, this SAS program follows the same steps as the [SAS program for the CDC growth charts](https://www.cdc.gov/growth-chart-training/hcp/computer-programs/sas.html). Additional details about the ages for which the various z-scores and percentiles are calculated are in Table 2 below.

The SAS program calculates z-scores and percentiles based on reference values in [WHOref_d.sas7bdat](https://www.cdc.gov/growth-chart-training/media/files/WHOref_d.sas7bdat). This reference dataset combines values from several [WHO datasets](https://www.who.int/toolkits/child-growth-standards/software). If you are not using SAS, you can download [WHOref_d.csv](https://www.cdc.gov/growth-chart-training/media/files/WHOref_d.csv) and create a program based on [who-source-code.sas](https://www.cdc.gov/growth-chart-training/media/files/who-source-code.sas) to do the necessary calculations.

> [!NOTE]
> **Reminder**: For children from birth to 2 years, use WHO growth charts. For children 2 to 20 years, use [CDC growth charts](https://www.cdc.gov/growthcharts/cdc-growth-charts.htm).

## Instructions

**Step 1**: Download the SAS program ([who-source-code.sas [SAS-6KB]](https://www.cdc.gov/growth-chart-training/media/files/who-source-code.sas)) and the reference data file ([WHOref_d.sas7bdat](https://www.cdc.gov/growth-chart-training/media/files/WHOref_d.sas7bdat)). Do not alter these files. Move them to a directory (folder) that SAS can access. If you are using Chrome or Firefox, right click to save the `who-source-code.sas` file.

For the example provided in Table 1, the files have been saved in: `c:\sas\growth charts\who\data`

**Step 2**: Create a `libname` statement in your SAS program to point at the folder location of 'WHOref_d.sas7bdat'. An example would be:

```sas
libname refdir ‘c:\sas\growth charts\who\data’;
```
Note that the SAS code expects this name to be `refdir`. Do not change this name.

**Step 3**: Set your existing dataset containing height, weight, sex, age, and other variables into a temporary dataset named `mydata`. Rename and code variables in your dataset as follows:

### Table 1: Instructions for SAS users (step 3), guidance on renaming and coding variables in your dataset.

Variable | Description
---------|------------
agedays  | Child’s age in days; must be present. If this value is not an integer, the program rounds to the nearest whole number. If age is known only to the completed number of weeks (for example, 5 weeks of age would represent any number of days between 35 and 41), multiply by 7 and consider adding 4 (median number of days in a week). If age is known only to the completed number of months, multiply by 365.25/12, and consider adding 15.
sex      | Coded as 1 for boys and 2 for girls.
height   | Recumbent length in cm. If standing height (rather than recumbent length) was recorded, add 0.7 cm to the values  For more information, see [WHO Child Growth Standards based on length/height, weight and age](https://pubmed.ncbi.nlm.nih.gov/16817681/)
weight   | Weight (kg)
bmi      | BMI (weight (kg) / height (m)²). If your data do not contain BMI, the program calculates it. If BMI is present in your data, the program will not overwrite it.
headcir  | Head circumference (cm)
armcir   | Arm circumference (cm)
tsf      | Triceps skinfold thickness (mm)
ssf      | Subscapular skinfold thickness (mm)

Percentiles and Z-scores for variables that are not in `mydata` will be coded as missing (`.`) in the output dataset (named `_whodata`). Sex (coded as 1 for boys and 2 for girls) and `agedays` must be in `mydata`. SAS code is unlikely to overwrite other variables in your dataset. Avoid having variable names that begin with an underscore, such as `_bmi`.

**Step 4**: Copy and paste the following line into your SAS program after the line (or lines) in step 3.

```sas
%include ‘c:\sas\growth charts\who\data\WHO-source-code.sas’; run;
```

If necessary, change this statement to point at the folder containing the downloaded '`WHO-source-code.sas`' file. This tells your SAS program to run the statements in '`WHO-source-code.sas`'.

**Step 5**: Submit the `%include` statement. This will create a dataset named `_whodata`. This dataset will contain all of your original variables, along with percentiles, z-scores, and flags for extreme values. The names and descriptions of these new variables in `_whodata` are in Table 2.

### Table 2: Z-Scores, percentiles, and extreme values (biologically implausible, BIV) in output dataset, `_whodata`
| Description | Percentile | Z-score | Flag for Extreme Values | Cutoff for Extreme Z-scores Low (Flag = -1) | Cutoff for Extreme Z-scores High (Flag = +1) |
|-------------|------------|---------|-------------------------|-------------|----------------|
| Weight-for-age for children aged from 0 to 1856 days | wapct | waz | _bivwt | < -6 | >=5 |
| Height-for-age for children aged from 0 to 1856 days | hapct | haz | _bivht | < -6 | >=6 |
| Weight-for-height for children with heights from 45 to 121 cm (these heights approximately correspond to ages 0 to 6 years) | whpct | whz | _bivwh | < -5 | >5 |
| BMI-for-age for children aged 0 to 1856 days | bmipct | bmiz | _bivbmi | < -5 | >5 |
| Head circumference-for-age for children aged from 0 to 1856 days | headcpct | headcz | _bivhc | < -5 | >5 |
| Arm circumference-for-age for children aged 91 to 1856 days| armcpct | armcz | _bivac | <-5 | >5 |
| Triceps skinfold thickness-for-age for children aged 91 to 1856 days|tsfpct | tsfz | _bivtsf | <-5 | >5 |
| Subscapular skinfold thickness-for-age for children aged 91 to 1856 days| ssfpct | ssfz | _bivssf | <-5 | >5 |

> [!IMPORTANT]
> Table 2 has been updated to include the actual variables from `_whodata`. The original table contained variables not present in `_whodata` (such as those prefixed with `mod_`) and lacked variables for arm circumference, triceps skinfold thickness, and subscapular skinfold thickness.

> [!NOTE]
> In addtion to the above variables, `_bivlow` and `_bivhigh` are included in `_whodata`. If any of the measurements (`_bivwt`, `_bivht`, `_bivwh`, `_bivbmi`,`_bivhc`,  `_bivac`, `_bivtsf`, or `_bivssf`) is -1, then `_bivlow` is 1; otherwise, `_bivlow` is 0.
> If any of those measurements is 1, then `_bivhigh` is 1; otherwise, `_bivhigh` is 0.

**Step 6**: Examine the new dataset, `_whodata`, with `PROC MEANS` or some other procedure to verify that the z-scores and other variables have been created. If a variable in Table 1 was not in your original dataset (for example, arm circumference), the output dataset will indicate that all values for the percentiles and z-scores of this variable are missing. If values for other variables are unexpectedly missing, make sure that you've renamed and recoded variables as indicated in Table 1 and that your SAS dataset is named mydata. The program should not modify your original data but will add new variables to your original dataset.

Sample SAS code corresponding to steps 2 to 6. You can cut and paste these lines into a SAS program, but you'll need to change the `libname` and `%include` statements to point at the folders containing the downloaded files.

```sh
c:\sas\growth charts\who\data
```

### Additional Information

Z-scores are calculated as $Z = \frac{ \left( \frac{\text{value}}{M} \right)^L - 1 }{ S \times L }$, in which "value" is a variable such as the child's BMI, weight, or height. The $L$, $M$, and $S$ values are in `WHOref_d.sas7bdat` and vary according to the child's sex and age or according to the child's sex and height. Percentiles are then calculated from the z-scores (for example, a z-score of 1.96 is equal to the 97.5 percentile). See more information on the [LMS method](https://pmc.ncbi.nlm.nih.gov/articles/PMC27365/).

### Extreme or biologically implausible values

The SAS code also flags extreme values (biologically implausible values, or BIVs) according to the [WHO criteria](https://www.who.int/toolkits/child-growth-standards/software). Each variable has a BIV flag coded as `-1` (an extremely low z-score), `+1` (extremely high z-score), or `0` (the z-score is between the low and high cut-points). These BIV flags, along with other variables in the output dataset `_whodata`, are shown in Table 2.

The z-scores in the output data set, `_whodata`, can also be used to construct other cut-points for extreme (or biologically implausible) values. For example, if the distribution of weight in your data is strongly skewed to the right, you might use `bmiz > 7` (rather than `bmiz > 5`) as the cut-point for extremely high BMI-for-age. This could be recoded as:
```sas
if -5 <= bmiz <= 7 then _bivbmi=0; *plausible; else if bmiz > 7 then _bivbmi=1; *high BIV; else if . < bmiz < -5 then _bivbmi= -1; *low BIV;
```

There are also two overall indicators of extreme values in the output dataset: `_bivlow` and `_bivhigh`. These variables indicate whether any measurement is extremely high (`_bivhigh=1`) or extremely low (`_bivlow=1`). If a child does not have an extreme value for any measurement, both variables are coded as `0`.

