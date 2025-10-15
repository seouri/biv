*************************************************************
* This sas program calculates percentiles and z-scores
* based on the WHO child growht standards
* (http://www.who.int/childgrowth/en/).
* It also flags biologically implausible values

* Unless you have a very good reason to alter this file (and this is unlikely),
* please do NOT make changes.  This file is meant to be called with a %include
* statement from your SAS program.
**************************************************************;

data _mydata _windata _wold; set mydata;
	_id=_n_; u=uniform(0); _agedays=round(agedays,1);  output _mydata;
	if bmi<=0 & weight>0 & height>0 then bmi=weight/((height/100)**2);
	if _agedays ge 731 then output _wold; 
		else output _windata; *only do calcs for kids who are 0 to 2 y of age;

*** 1. IMPORT THE CSV FILE ***;
PROC IMPORT DATAFILE="&dirpath/WHOref_d.csv"
    OUT=refdir.whoref_d_csv
    DBMS=CSV
    REPLACE;
    GETNAMES=YES;
    GUESSINGROWS=MAX;
RUN;

*** 2. CREATE whoref_d DATASET WITH CORRECT VARIABLE TYPES ***;
data refdir.whoref_d;
    set refdir.whoref_d_csv;
***	sex = input(sex, best8.);***;
	_agedays = input(_agedays, best32.);
	_bmi_l = input(_bmi_l, best32.);
	_bmi_m = input(_bmi_m, best32.);
	_bmi_s = input(_bmi_s, best32.);
	_tsf_l = input(_tsf_l, best32.);
	_tsf_m = input(_tsf_m, best32.);
	_tsf_s = input(_tsf_s, best32.);
	_ssf_l = input(_ssf_l, best32.);
	_ssf_m = input(_ssf_m, best32.);
	_ssf_s = input(_ssf_s, best32.);
	_armc_l = input(_armc_l, best32.);
	_armc_m = input(_armc_m, best32.);
	_armc_s = input(_armc_s, best32.);
	_headc_l = input(_headc_l, best32.);
	_headc_m = input(_headc_m, best32.);
	_headc_s = input(_headc_s, best32.);
	_wei_l = input(_wei_l, best32.);
	_wei_m = input(_wei_m, best32.);
	_wei_s = input(_wei_s, best32.);
	_len_l = input(_len_l, best32.);
	_len_m = input(_len_m, best32.);
	_len_s = input(_len_s, best32.);
	_len = input(_len, best32.);
	_wfl_l = input(_wfl_l, best32.);
	_wfl_m = input(_wfl_m, best32.);
	_wfl_s = input(_wfl_s, best32.);
run;

***********************************************************************;
*** END MODIFICATION
***********************************************************************;

data _wref1; set refdir.whoref_d; *** refers to refdir;
	if _denom='forage' & .< _agedays<731; u=uniform(0); *limit to 2 y of age and under;

	
proc sort data=_wref1; by sex _agedays; proc sort data=_windata; by sex _agedays;
data _wt1; merge _wref1(in=in1) _windata(in=in2); by sex _agedays; if in2; 

*********** calculate z-scores *********;
data _wt1; set _wt1; 
	array a height headcir bmi armcir ssf tsf weight; 
		do over a; if a LE 0 then a=.;  end; *in case someone has a negative or 0 value;

	haz=(((height/_len_M)**_len_L)-1)/(_len_S*_len_L); 	hapct=100*probnorm(haz);
	headcz=(((headcir /_headc_M)**_headc_L)-1)/(_headc_S*_headc_L);  headcpct=100*probnorm(headcz); 
	*head circ and height z's (for age) do not get adusted for SD2\3 distance; 

	array value  	bmi 		armcir		ssf 		 tsf 	    weight; 
	array L	 		_bmi_L 		_armc_L		_ssf_L 		_tsf_L 		_wei_L; 
	array M 		_bmi_M	 	_armc_M		_ssf_M 		_tsf_M 		_wei_M; 
	array S 		_bmi_S 		_armc_S		_ssf_S	 	_tsf_S 		_wei_S; 
	array Z 		bmiz 		armcz		ssfz 		tsfz 		waz; 
	array P 		bmipct 		armcpct		ssfpct 		tsfpct 		wapct;
	do over value; 
		if value>0 then Z=(((value/M)**L)-1)/(S*L);  **someone may have entered 0 by mistake;
			*if  abs(L) <0.01 then Z0=LOG(value/M)/S;  *WHO doesn't use this adjustment;	 

		if abs(z) ge 3 then do; ** adjust extreme z scores using distance between SD2 and SD3; 
			sd2pos=M*(1+L*S*2)**(1/L); sd2neg=M*((1+L*S*(-2))**(1/L));
	    	sd3pos=M*(1+L*S*3)**(1/L); sd3neg=M*((1+L*S*(-3))**(1/L));
			sd23pos= sd3pos - sd2pos;  sd23neg=  sd2neg - sd3neg; 
				if Z ge 3 then Z= 3 + (value - sd3pos)/sd23pos; 
					else if Z le -3 & z > . then Z= -3 + (value - sd3neg)/sd23neg;
			end;
   	P=100*probnorm(Z);
	end;

******** weight for height/length calculations *********;

data _windata2; set _windata (keep=_id sex weight height _agedays); 
	if height >.; height=round(height,.01);

data _wref2; set refdir.whoref_d (keep=sex _denom _len _wfl_l--_wfl_s rename=(_len=height)); 
	where _denom='forlen';   *var names are _len and _wfl_l/m/s in reference data; 
		
proc sort data=_wref2; by sex height; proc sort data=_windata2; by sex height; run;

data _wt2; merge _windata2(in=in1) _wref2(in=in2); by sex height; if in1; 
 	if weight >.z then whz= (((weight / _wfl_M)**_wfl_L)-1) / (_wfl_S * _wfl_L);  
	
	if abs(whz) ge 3 then do;
		Sd2pos=_wfl_M*(1+_wfl_L*_wfl_S*2)**(1/_wfl_L); Sd2neg=_wfl_M*((1+_wfl_L*_wfl_S*(-2))**(1/_wfl_L));
	    Sd3pos=_wfl_M*(1+_wfl_L*_wfl_S*3)**(1/_wfl_L); Sd3neg=_wfl_M*((1+_wfl_L*_wfl_S*(-3))**(1/_wfl_L));
		Sd23pos= Sd3pos - Sd2poS;  Sd23neg= Sd2neg - Sd3neg; 
			if whz ge 3 then whz= 3 + (weight - sd3pos)/sd23pos; 	
				else if whz le -3 & whz > . then whz= -3 + (weight - sd3neg)/sd23neg;
		end;
   	whpct=100*probnorm(whz);

proc sort data=_wt1; by _id; proc sort data=_wt2; by _id; run;

data _wd1; merge _wt1(in=in1) _wt2(keep=_id sex _agedays whz _wfl_l _wfl_m _wfl_s whpct); by _id; if in1;
		
	if haz gt .z then do; _bivht=0; if haz < -6 then _bivht= -1; if haz>= 6 then _bivht=1; end;
	if waz gt .z then do; _bivwt=0; if waz< -6 then _bivwt=-1; if waz>= 5 then _bivwt=1; end;
		
	array x1  bmiz    armcz  headcz    ssfz    tsfz     whz;
	array x2 _bivbmi _bivac  _bivhc   _bivssf  _bivtsf _bivwh;
	do over x1; 
		if x1> .z then do;
			x2=0; if x1>5 then x2=1; if .z<x1< -5 then x2= -1; 
		end;
	end; 

	min= min(of _bivht _bivwt _bivbmi _bivac _bivhc _bivssf _bivtsf _bivwh);
		if min>=0 then _bivlow=0; else if min= -1 then _bivlow=1;
	max= max(of _bivht _bivwt _bivbmi _bivac _bivhc _bivssf _bivtsf _bivwh);
		if max=0 or max= -1 then _bivhigh=0; else if max= 1 then _bivhigh=1;	

data _woutdata; set _wd1 _wold; *combine with older kids who have been excluded from z-score calcs;
	keep _bivac _bivbmi _bivhc _bivhigh _bivht _bivlow _bivssf _bivtsf _bivwh _bivwt _id whz whpct 
	     agedays armcpct armcz bmi bmipct bmiz hapct haz headcpct headcz sex ssfpct ssfz tsfpct tsfz wapct waz;
	
	label waz='weight-for-age Z'  wapct='weight-for-age percentile'  bmiz='BMI-for-age Z'  bmipct='BMI-for-age percentile' 
		haz='height-for-age Z'  hapct='height-for-age percentile'  whz='weight-for-height Z'   
		_bivwt='BIV weight-for-age'  headcz='head_circ-for-age Z' headcpct='head_circ-for age perc' 	_bivhc='BIV head_circ'  
		_bivbmi='BIV BMI-for-age' whpct='weight-for-height percentile' 	_bivht='BIV height-for-age' 
		_bivwh='BIV weight-for-height'  _bivlow='any low BIV'  _bivhigh='any high BIV' tsfz='triceps skinfold z'
		tsfpct='triceps skinfold percentile' ssfz='subscap skinfold z' ssfpct='subscap skinfold percentile'
		armcz='arm circumference z' armcpct='arm circumference percentile' _bivac='BIV arm circum' _bivssf='BIV subscap skinfold'
		_bivtsf='BIV triceps skinfold';

proc sort data=_woutdata; by _id; proc sort data=_mydata; by _id;
data _whodata; update _woutdata  _mydata; by _id; 
data _whodata; set _whodata; drop _id _agedays u;

