** LINK TO OUR GITHUB 
** https://github.com/Staiana/Greenwashing_Project_Group_6/tree/main
** We used python for cleaning our data and for the descriptive analysis 

* Set the working environment (Change to your directory!!) * /Users/carmelaperri/Desktop/REME
cd "\Users\simta\OneDrive\Documents\documents importants\Université\HEC3\Recherches"

clear
set more off

* Install necessary packages
ssc install drdid, replace
ssc install csdid, replace
ssc install outreg2, replace
ssc install estout, replace

* Load the data
import delimited "final_df.csv", clear

* Describe the data to understand the variable structures
describe

* Convert string dates to Stata's date format for accurate processing
gen event_date = date(accusation_date, "DMY")
format event_date %td
gen observation_date = date(date, "DMY")
format observation_date %td





* Encoding string variables to numeric IDs for regression analysis
encode ticker, gen(ticker_id)
foreach day in friday thursday wednesday tuesday {
    encode day_`day', gen(new_day_`day')
}
foreach sector in consumercyclical consumerdefensive communicationservices energy financialservices healthcare industrials realestate technology utilities {
    encode sector_`sector', gen(new_sector_`sector')
}


forvalues i = 30(-1)0 {
    rename days_from_accusation_`i' days_from_accusation_`i'_before
}

local num 1
forvalues i = 65(1)93 {
    rename v`i' days_from_accusation_`num'
    local num = `num' + 1
}


* Creating variables essential for DiD analysis
gen treated = observation_date >= event_date
*by ticker_id (observation_date), sort: gen post = observation_date >= event_date if _n == 1
*replace post = post[_n-1] if post[_n-1] == 1
gen event_time = year(observation_date) - year(event_date)
label variable event_time "Years since accusation"


label variable days_from_accusation_28_before "-28"
label variable days_from_accusation_14_before "-14"
label variable days_from_accusation_7_before "-7"
label variable days_from_accusation_3_before "-3"
label variable days_from_accusation_1_before "-1"
label variable days_from_accusation_0 "0"
label variable days_from_accusation_1 "1"
label variable days_from_accusation_2 "2"
label variable days_from_accusation_3 "3"
label variable days_from_accusation_5 "5"
label variable days_from_accusation_7 "7"
label variable days_from_accusation_10 "10"
label variable days_from_accusation_14 "14"
label variable days_from_accusation_20 "20"
label variable days_from_accusation_28 "28"

label variable logclose "Close Price"
label variable volume "Volume"
label variable dividends "Dividends"
label variable stock_split "Stock Split"
label variable new_day_friday "Friday"
label variable new_day_thursday "Thursday"
label variable new_day_tuesday "Tuesday"
label variable new_day_wednesday "Wednesday"

label variable treated "Accusation"

drop accusation_date
drop date
drop day_friday
drop day_thursday
drop day_tuesday
drop day_wednesday
drop v30
drop v31
drop v32
drop v33


* Summary statistics to inspect the data
summarize

* Setting up panel data structure
xtset ticker_id observation_date
xtdes

*************************************************
*************** DIFFERENCE-IN-DIFFERENCE ***************
*************************************************

* Regression model with Fixed Effects 
xtdidregress (logclose days_from_accusation_*) (treated), g(ticker_id) t(observation_date) vce(cluster ticker_id)

estimates store Model2


* Regression model with Fixed Effects and controls
xtdidregress (logclose volume dividends stock_split days_from_accusation_*) (treated), g(ticker_id) t(observation_date) vce(cluster ticker_id)

estimates store Model3


* Extended xtdidregress command including more fixed effects and covariates
xtdidregress (logclose volume dividends stock_split days_from_accusation_* new_day_friday new_day_thursday new_day_tuesday new_day_wednesday i.year i.month) ///
 (treated), g(ticker_id) t(observation_date) vce(cluster ticker_id) 

estimates store Model5


esttab Model2 Model3 Model5 using "table_did.xls", se title("Table 3") mtitles("FE" "FE and Controls" "Extended Model") label replace star(* 0.05 ** 0.025 *** 0.01) 

esttab Model2 Model3 Model5 using "table_did.rtf", se title("Table 3") mtitles("FE" "FE and Controls" "Extended Model") label replace star(* 0.05 ** 0.025 *** 0.01) 

esttab Model2 Model3 Model5 using "table_did.tex", se title("Table 3") mtitles("FE" "FE and Controls" "Extended Model") label replace star(* 0.05 ** 0.025 *** 0.01) 



****************************************
************ Event plot ****************
****************************************

ssc install coefplot
* Plotting coefficients using coefplot, offsetting models for clarity


coefplot (Model2, offset(-0.1) mcolor(navy) lcolor(navy) ciopts(lcolor(navy))) ///
         (Model5, offset(0.1) mcolor(green) lcolor(green) ciopts(lcolor(green))), ///
    vertical ///
    keep(days_from_accusation_28_before days_from_accusation_14_before days_from_accusation_7_before days_from_accusation_3_before days_from_accusation_1_before days_from_accusation_0 days_from_accusation_1 days_from_accusation_2 days_from_accusation_3 days_from_accusation_5 days_from_accusation_7 days_from_accusation_10 days_from_accusation_14 days_from_accusation_20 days_from_accusation_28) /// 
    recast(ci) ///
	yline(0) ///
	xline(5.5) ///

    citop ///
    graphregion(color(white)) ///
    plotregion(color(white)) ///
    name(EventWindowGraph, replace) ///
    title("Event Window Analysis: Model Comparisons", size(medium)) ///
	xtitle("Days around accusation") ytitle("Coefficients") ///
	legend(order(1 "Fixed Effects" 3 "Extended Model") ///
           rows(1) size(small) symxsize(*0.6) symysize(*0.6) ///
           col(1) position(12) ring(0))
		   
 graph export "graph2.png", replace
