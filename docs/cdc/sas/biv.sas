/* Define macro variable for the directory path */
%let dirpath = /home/u64365754/biv;
libname refdir "&dirpath";

/* Create dataset with all 32 test cases */
data mydata;
  informat agemos best8. sex best2. weight height bmi headcir best8.2;
  input agemos sex weight height bmi headcir;
  agedays = agemos * (365.25 / 12); /* Calculate agedays for WHO */
  datalines;
0.5 1 3.5 50.0 . .
3.0 2 6.0 60.0 . 40.0
6.0 1 8.5 68.0 . 43.5
12.0 2 4.5 65.0 . 37.0
12.0 1 10.0 74.0 . 45.5
18.0 2 13.5 81.0 . 48.0
23.0 1 11.0 84.0 . 46.5
24.0 1 12.5 87.0 16.5 47.5
72.0 2 22.0 118.0 15.7 51.0
132.0 1 40.0 145.0 19.0 .
192.0 2 55.0 162.0 21.0 .
240.0 1 70.0 175.0 22.8 .
48.0 1 40.0 102.0 38.2 50.0
60.0 2 25.0 155.0 10.4 48.0
72.0 1 12.0 105.0 10.9 51.0
84.0 2 45.0 165.0 16.5 .
120.0 1 25.0 130.0 14.8 .
168.0 2 80.0 170.0 27.7 .
30.0 1 8.0 91.0 9.7 .
144.0 2 95.0 175.0 31.0 .
42.0 1 10.0 85.0 13.8 43.0
50.5 1 20.0 105.0 18.1 50.5
92.5 2 40.0 128.0 24.4 .
114.5 1 45.0 134.0 25.0 .
24.0 1 12.5 87.0 16.5 47.5
23.99 2 12.0 86.5 . 47.0
241.0 1 75.0 180.0 23.1 .
36.0 2 14.0 95.0 15.5 42.0
96.0 1 35.0 130.0 20.7 .
8.0 1 2.5 60.0 . 38.0
12.0 2 12.0 75.0 . 50.0
156.0 1 90.0 170.0 31.1 .
216.0 2 110.0 160.0 43.0 .
;
run;

/* Include the CDC growth charts program */
%include "&dirpath/cdc-source-code-csv.sas"; 
run;

/* Export _cdcdata to CSV for external analysis */
/* Create a new dataset with columns in the specified order using retain */
data _cdcdata_export;
  keep agemos sex weight height bmi headcir 
       waz wapct _bivwt 
       haz hapct _bivht 
       bmiz bmipct _bivbmi
       headcz headcpct _bivhc
       whz whpct _bivwh 
       mod_waz mod_haz mod_bmiz mod_headcz mod_whz;
  retain agemos sex weight height bmi headcir 
         waz wapct _bivwt 
         haz hapct _bivht 
         bmiz bmipct _bivbmi
         headcz headcpct _bivhc
         whz whpct _bivwh 
         mod_waz mod_haz mod_bmiz mod_headcz mod_whz;
  set _cdcdata;
run;

/* Export the dataset to CSV */
proc export data=_cdcdata_export
  outfile="&dirpath/biv_sas_cdc.csv"
  dbms=csv
  replace;
run;

/* Include the WHO growth charts program */
%include "&dirpath/who-source-code-csv.sas"; 
run;

/* Export _whodata to CSV for external analysis */
/* Create a new dataset with only the specified columns in the desired order */
data _whodata_export;
  keep agemos sex weight height bmi headcir
       waz wapct _bivwt
       haz hapct _bivht
       bmiz bmipct _bivbmi
       headcz headcpct _bivhc
       whz whpct _bivwh;
  retain agemos sex weight height bmi headcir
         waz wapct _bivwt
         haz hapct _bivht
         bmiz bmipct _bivbmi
         headcz headcpct _bivhc
         whz whpct _bivwh;
  set _whodata;
run;

/* Export the dataset to CSV */
proc export data=_whodata_export
  outfile="&dirpath/biv_sas_who.csv"
  dbms=csv
  replace;
run;
