# 1. Install and load packages

## The pacman package was used to install and load all the packages required for analysis.

install.packages("pacman")
library(pacman)
p_load(tidyverse, styler, gt, GGally, ggplotify)
p_load(VIM, naniar, visdat, simputation, mice, missForest, ggmice,
ggcorrplot, miceForest)
p_load(pROC)
p_load(rmarkdown)
p_load(tinytex)
p_load(recipes, caret, adabag, xgboost, nnet, glmnet, kknn,ParBayesianOptimization)
p_load(caTools)
p_load(foreach, doParallel)
p_load(gtsummary)
p_load(visdat)
p_load(DataExplorer)
p_load(tinytex)
detectCores()

## Register three cores for parallel computing to make machine learning modelling more efficient.

registerDoParallel(cores = 3)

# 2. Import datasets to working environment

## All available datasets were imported into the Rstudio working environment using the read_csv function from the dplyr package.

Cohort <- read_csv("C:/...")
Demography <- read_csv("C:/...")
Region_CRC <- read_csv("C:/...")
Region_Practice <- read_csv("C:/...")
Strategy <- read_csv("C:/...")
Distance <- read_csv("C:/...")
Practice_Dem <- read_csv("C:/...")
HACE_13_14_GGC_Tay <- read_csv("C:/...")
HACE_15_16_GGC <- read_csv("C:/...")
HACE_15_16_Lanarkshire <- read_csv("C:/...")

# 3. Data Preparation

# 3.1 Health and Care Experience Dataset

#The Health and Care Experience (HACE) survey was requested over two different year groups
#(13/14 and 15/16) as the data for 1st invitation of patients to the trial occurred over different year
#groups for Greater Glasgow and Clyde (GG&C) practices. To be most accurate, the survey year
#needs to be matched with the year that the patient was invited to the trial, dtFirstContact, since
#this will be more relevant to the patient deciding to participate if satisfaction with the practice is
#relevant. NB: Tayside practices were invited during the year 13/14 only, Lanarkshire practices
#were 15/16 only. The 13/14 GG&C survey data was provided along with the 13/14 survey data
#for Tayside, HACE_13_14_GGC_Tay. The following shows the steps taken to accurately match
#survey year to date of 1st contact.

# Change character variable to numeric for HACE_15_16_GGC percentage positive

HACE_15_16_GGC <- HACE_15_16_GGC %>%
mutate_at("Percentage Positive", as.numeric)

# Make Cohort_HACE dataset for linking HACE data by year of date 1st contacted.

Cohort_HACE <- Cohort %>%
select(c(ProPracticeID, dtFirstContact))

# Convert date format to year only for matching purposes

Cohort_HACE$dtFirstContact <- format(
    as.POSIXct(Cohort_HACE$dtFirstContact,
    format = "%d/%m/%Y %H:%M"
    ),
    format = "%Y"
    )

# Label HACE survey datasets with year for the purpose of linking to year of 1st contact. For naming 
#purposes only, survey year 13 is used instead of 13/14 and 15 instead of 15/16

HACE_13_14_GGC_Tay <- mutate(HACE_13_14_GGC_Tay, "Survey_Year" = "13")
HACE_15_16_GGC <- mutate(HACE_15_16_GGC, "Survey_Year" = "15")
HACE_15_16_Lanarkshire <- mutate(HACE_15_16_Lanarkshire, "Survey_Year" = "15")

# Bind rows from all HACE survey datasets to create one dataset with all survey results

HACE_all <- bind_rows(HACE_13_14_GGC_Tay, HACE_15_16_GGC, HACE_15_16_Lanarkshire)

# Replace spaces in column names with underscores

colnames(HACE_all) <- gsub(" ", "_", colnames(HACE_all))

# Give HACE practice variable name the same as others for joining ProPracticeID

HACE_all <- rename(HACE_all, ProPracticeID = AnonPrac)

# Join prepared Cohort_HACE dataset with HACE_all dataset using "ProPracticeID" as matching key and 
# specifying a many-to-many relationship due to duplicate practices in HACE_all caused by
# multiple survey years for the same practice

Cohort_HACE <- left_join(Cohort_HACE, HACE_all, by = "ProPracticeID", 
    relationship = "many-to-many")

# Convert Survey_Year and dtFirstContact to numeric for purpose of applying logical operators

Cohort_HACE$Survey_Year <- as.numeric(Cohort_HACE$Survey_Year)
Cohort_HACE$dtFirstContact <- as.numeric(Cohort_HACE$dtFirstContact)

# Create Lanarkshire and Tayside HACE since it is only GG&C whose recruitment was staggered 
#over 2013/2014 and 2015/2016
# Deduplicated following many-to-many relationship during linkage

Cohort_HACE_Lanarkshire_Tayside <- Cohort_HACE %>%
filter(Health_Board_Name %in% c("NHS Lanarkshire", "NHS Tayside")) %>%
distinct(ProPracticeID, .keep_all = TRUE)

# Create GG&C HACE dataset

Cohort_HACE_GGC <- Cohort_HACE %>%
filter(!Health_Board_Name %in% c("NHS Lanarkshire", "NHS Tayside"))

# Match observations where year of recruitment is the same as survey year

Cohort_HACE_GGC_13_14 <- Cohort_HACE_GGC %>%
filter((dtFirstContact == 2013 | dtFirstContact == 2014) & Survey_Year == 13) %>%
distinct(ProPracticeID, .keep_all = TRUE)
Cohort_HACE_GGC_15_16 <- Cohort_HACE_GGC %>%
filter((dtFirstContact == 2015 | dtFirstContact == 2016) & Survey_Year == 15) %>%
distinct(ProPracticeID, .keep_all = TRUE)

# Combine all HACE cleaned datasets

HACE_clean <- bind_rows(Cohort_HACE_GGC_13_14, Cohort_HACE_GGC_15_16, Cohort_HACE_Lanarkshire_Tayside)

# remove redundant matching variables

HACE_clean <- HACE_clean %>%
select(-c(dtFirstContact, Survey_Year))

# 3.2. Recruited dataset

# remove non-GP invites from recruitment source
GP_recruited <- filter(Strategy, GP_Invite == "Yes")

# 4. Data Linkage

#The denominator datasets contain data #on all 77,746 individuals who were invited to the trial. 
#This includes those who were recruited.
#The numerator datasets contain data on only the 10,176 patients that were recruited.

# 4.1 Numerator dataset linkage

#recruitment_all contains the two datasets that contain data only for those that were recruited.
#Initially, a left_join is made between the GP_recruited dataset – all those recruited – and
#the Region_CRC dataset – the regional trial centres where the patients were recruited to.

# combine recruited datasets with PROCHI as primary key
recruitment_all <- GP_recruited %>%
left_join(Region_CRC, by = "PROCHI")

# 4.2 Denominator dataset linkage

# invited_all is the complete denominator dataset and is made by sequentially linking numerator
# datasets with the left_join function.

# combine denominator datasets
invited_all <- left_join(Cohort, Demography, by = "PROCHI") %>%
left_join(Distance, by = "PROCHI") %>%
left_join(Practice_Dem, by = "ProPracticeID") %>%
left_join(HACE_clean, by = "ProPracticeID")

# 4.3 Full dataset linkage

#The complete dataset is given the object name, patient_full. This is made using a
#full_join of invited_all and recruitment_all, i.e., the denominator and numerator
#datasets respectively. A full join is required because all observations should be kept.

# combine numerator and denominator datasets

patient_full <- full_join(invited_all, recruitment_all, by = "PROCHI")

# 5. Renaming columns

#Columns are renamed with lower case and underscore convention to separate words.

# Rename variables
patient_full <- patient_full %>%
rename(
prochi = PROCHI, propracticeid = ProPracticeID, ur2fold = UR2FOLD,
ur3fold = UR3FOLD, ur6fold = UR6FOLD, ur8fold = UR8FOLD, IMD_year
= IMD_YEAR, IMD_type = IMD_Type, census_year = Census_Year,
age_range = AgeRange, date_registered = Date_Registered,
date_1st_contact = dtFirstContact, overall_decile = Overall_DECILE,
income_decile = Income_DECILE, employment_decile = Employment_DECILE,
health_decile = Health_DECILE, education_decile = EducationSkillsTraining_DECILE,
access_to_services_decile = AccessToServices_DECILE, housing_decile =
Housing_DECILE, crime_decile = Crime_DECILE, health_board = Cohort_hb, 
route_distance = Route_Distance, route_time = Route_Time, 
direct_distance = Direct_Distance, NHS_board = NHS_Board,
GP_headcount = Number_of_GPs, urban_rural_category = Urban_Rural_category, 
most_deprived = Patients_in_15_most_deprived_areas,
practice_population = Total_Practice_Population, percent_positive =
Percentage_Positive, health_board_name = Health_Board_Name,
recruited = Recruitment_source, GP_invite = GP_Invite
)

# 6. Linked Dataset cleaning

#This section describes the steps taken to appropriately treat any data issues, including missing
#data, redundant variables and conversion of variable types for machine learning.

# 6.1. Recruited data

# Show number of individuals by GP_invite values
patient_full %>%
count(GP_invite)

# Show number of individuals by recruited values
patient_full %>%
count(recruited)

# Match recruited individuals between GP_invite and recruited variables
patient_full %>%
select(GP_invite, recruited) %>%
mutate(GP_invite = case_when(
GP_invite == "Community/workplace event" ~ "Yes",
GP_invite == "Media" ~ "Yes",
GP_invite == "Other" ~ "Yes",
GP_invite == "Word of Mouth" ~ "Yes",
TRUE ~ as.character(GP_invite)
)) %>%
mutate(recruited = case_when(recruited == "GP" ~ "Yes")) %>%
mutate(match_check = recruited == GP_invite) %>%
summarise(
total_matches = sum(match_check, na.rm = TRUE),
total_non_matches = sum(!match_check, na.rm = TRUE)
)

#This confirms that the variables recruited and GP_invite have the same total number of
#recruited individuals and that these match across datasets. recruited is used since these
#patients were invited to the study through their GP practice as well as having been exposed to
#other sources of recruitment material.

# 6.2 Variable Selection

#Several datasets had overlapping variables. There are also singleton variables that only have one
#observation value. There is therefore a need to remove variables that do not add information
#since this is redundant and add to computational demand.

# Singleton variables

#It is known that for IMD_type, IMD_year and census_year there is
#only one observation value. This is confirmed below.

# Show distinct values of observations for IMD_type, IMD_year and census_year
patient_full %>%
select(IMD_type, IMD_year, census_year) %>%
distinct()

#This confirms that IMD_type, IMD_year and census_year have only one observation value.
#These three variables are therefore removed from the linked dataset.

#Health Boards
#There were 3 health board variables from different datasets and with different
#numbers of missing data. The amount of missingness was counted for all three.

# Show total NA for health_board_name, NHS_board and health_board
patient_full %>%
select(health_board_name, NHS_board, health_board) %>%
is.na() %>%
colSums()

# filter the 1 row that contains the NA for health_board
patient_full %>%
filter(is.na(health_board))

#This missing health_board value relates to an individual who has no completed fields other
#than that they were recruited. They are removed for this reason and represent the only row that is
#removed from the dataset.

#Removing redundant variables 

#The variables described above are removed in the following code.

patient_full <- patient_full %>%
select(-c(GP_invite, IMD_year, IMD_type, census_year, CRC_HB,
health_board_name, NHS_board))

#Remove UR 2,3 and 6-fold as 8-fold is available and has most detail.

patient_full <- patient_full %>%
select(-c(ur2fold, ur3fold, ur6fold))

# 6.3 Variable type conversion

#There are several variables in patient_full that are character or factor variables. It may
#improve model performance to encode these as numerical values for the purpose of machine
#learning.

# 6.3.1 Time

#date_1st_contact is a character format and is in the format d/m/Y H:M. This is
#converted to the format d/m/Y and then converted to a numerical variable, representing the
#number of of days since the date of 1st contact for the purpose of modelling.

# Convert dtFirstContact from character to ymd_hms and remove H:M element
patient_full$date_1st_contact <- format(
as.POSIXct(patient_full$date_1st_contact,
format = "%d/%m/%Y %H:%M"
),
format = "%d/%m/%Y"
)

# convert string to date
patient_full$date_1st_contact <-
lubridate::dmy(patient_full$date_1st_contact)
# calculate the earliest date for both date variables
patient_full %>%
summarise(earliest_1st_contact = min(date_1st_contact, na.rm = TRUE),
earliest_registered = min(date_registered, na.rm = TRUE))

#The earliest date of these two variables is used to set the
#origin date when converting the date to numeric using the POSIXct function. The default origin
#date for POSIXct is the 01/01/1970 UTC. This will create a numeric variable that reports the
#number of days from the new origin date. This prevents large numbers if the origin date is set to
#01/01/1970.

#The next code chunk creates a vector with the origin date.

# convert to date object
origin <- as.Date(origin, format = "%d/%m/%Y")

# subtract origin date from the date registered and date of 1st contact
#variables and coerce this to a numeric value.
patient_full$date_1st_contact_num <-
as.numeric(patient_full$date_1st_contact - origin)
patient_full$date_registered_num <-
as.numeric(patient_full$date_registered - origin)

# create variable which is the difference between the date that the
#practice registered and the date of 1st invitation of the patient to
#the trial
patient_full <- patient_full %>%
mutate(date_diff = date_1st_contact_num - date_registered_num)

# remove original dates and reorder so that the outcome variable remains
# as the last column

patient_full <- patient_full %>%
select(-c(date_1st_contact, date_registered)) %>%
relocate(c(date_1st_contact_num, date_registered_num, date_diff),
.before = recruited)

# 6.3.2 Age
Age data is provided in quinquennia. For modelling purposes these are recoded to integers.

# Show unique age bands
unique(patient_full$age_range)

#count observations in each age range and convert to "<5" when n < 5
patient_full %>%
count(age_range) %>%
mutate(
across(n, ~
case_when(.x < 5 ~ "<5",
TRUE ~ as.character(.x)))
)

#Recode age bands as integers
patient_full$age_range <- as.numeric(gsub(
"10 - 14", 1,
gsub(
"15 - 19", 2,
gsub(
"20 - 24", 3,
gsub(
"25 - 29", 4,
gsub(
"30 - 34", 5,
gsub(
"35 - 39", 6,
gsub(
"40 - 44", 7,
gsub(
"45 - 49", 8,
gsub(
"50 - 54", 9,
gsub(
"55 - 59", 10,
gsub(
"60 - 64", 11,
gsub(
"65 - 69", 12,
gsub(
"70 - 74", 13,
gsub(
"75 - 79", 14,
gsub(
"80 - 84", 15,
gsub(
"85 - 89", 16,
gsub("90 - 94", 17,
patient_full$age_range)
)
)
)
)
)
)
)
)
)
)
)
)
)
)
)
))

# 6.3.3 Other variables

# convert sex from character to factor
patient_full$sex <- factor(patient_full$sex)

# convert Cohort_hb from character to factor
patient_full$health_board <- factor(patient_full$health_board)

# convert direct_distance from character to numeric
patient_full$direct_distance <- as.numeric(patient_full$direct_distance)

# parse numbers in urban_rural_category so that only numbers remain
# "Urban 1 - Large Urban Areas" "Urban 2 - Other Urban Areas"
# "Urban 3 - Accessible Small Towns" "Urban 4 - Remote Small Towns"
# "Urban 6 - Accessible Rural" "Urban 7 - Remote Rural"
patient_full$urban_rural_category <- parse_number(patient_full$urban_rural_category)

# convert recruited from numeric to factor
patient_full <- patient_full %>%
mutate_at("recruited", as.factor)

# 6.4. One-hot encoding

#sex and health_board are factor variables. These need to be encoded rather than only converted
#to numeric variable types, since they don’t have a natural order. If they were converted to a
#numeric variable, the ML algorithm would assign importance to the difference between numerical
#values.
#The caret package contains functions to encode variables. dummyVars, for example, implements
#one-hot encoding, which creates a new column for each category value and assigns a 1 or 0
#depending on whether that value is present or not.

# Create vector containing variables for encoding
encode_formula <- ~ sex + health_board

#Create a dummyVars object which contains the model to apply the encoding
dummyvar_model <- dummyVars(encode_formula, data = patient_full)

#Apply the encoding to patient_full
dummy_patient_full <- predict(dummyvar_model, patient_full)

#Remove the original sex and health_board variables from patient_full dataset
patient_full <- patient_full %>%
select(-c(sex, health_board))

#Combine columns from one-hot encoded dataset and patient_full
patient_full <- bind_cols(patient_full, dummy_patient_full)

#Rename variables to naming convention
patient_full <- patient_full %>%
rename(sex_F = sex.F, sex_M = sex.M, health_board_glasgow =
health_board.Glasgow, health_board_lanarkshire =
health_board.Lanarkshire, health_board_tayside =
health_board.Tayside)

#Reorder variables so that recruited remains as the last column
patient_full <- patient_full %>%
    relocate(health_board_glasgow, health_board_lanarkshire,
    health_board_tayside, starts_with("sex"), .before = recruited)

# 6.5 Observation name conversion

#Initial conversion changes character GP to numeric, 1. This represents those recruited to the trial.
#NAs in the recruited variable resulted from using full_join on GP_recruited numerator
#dataset with the denominator datasets. These are therefore converted to 0, representing those not
#recruited to the trial. Finally, for the purpose of using the machine learning package caret, the
#numeric variables 0 and 1 are converted to No and Yes respectively.

#Display the structure of the recruited variable
str(patient_full$recruited)

# convert character GP, to 1
patient_full$recruited <- as.numeric(gsub("GP", 1,
    patient_full$recruited))

# replace NAs with 0 since these represent missing data following linkage
# of numerator with denominator datasets
patient_full <- patient_full %>%
replace_na(list(recruited = 0))

# replace numbers 0 and 1 with No and Yes respectively. This is a
# requirement for use in the caret package
patient_full$recruited <- as.factor(gsub(
"1", "Yes",
gsub("0", "No", patient_full$recruited)
))

# display structure of converted variable
str(patient_full$recruited)

# 6.6 Variables prior to feature engineering

# Dataset variables following initial cleaning, prior to any feature engineering are:
names(patient_full)

# 7. Feature Engineering

# Additional features were created that may improve the predictive model accuracy.

# create general practitioner headcount to practice population ratio
patient_full <- patient_full %>%
mutate(gpcount_pop_ratio = practice_population / GP_headcount)

# Create male to female practice ratio
patient_full <- patient_full %>%
mutate(male_female_ratio = (M_0_4 + M_5_14 + M_15_24 + M_25_44 +
M_45_64 + M_65_74 + M_75_84 + M_85) /
(F_0_4 + F_5_14 + F_15_24 + F_25_44 + F_45_64 + F_65_74 + F_75_84 +
F_85))

# create age groupings for practice: younger, middle, older
patient_full <- mutate(patient_full,
younger_ratio = (M_0_4 + M_5_14 + M_15_24 + F_0_4 + F_5_14 + F_15_24) /
practice_population, middle_ratio = (M_25_44 + M_45_64 + F_25_44 +
F_45_64) / practice_population, older_ratio = (M_65_74 + M_75_84 + M_85 + F_65_74 + F_75_84 + F_85) /
practice_population
)

# create route speed from route time and distance
patient_full <- mutate(patient_full, route_speed = route_distance /
route_time)

# reorder so that recruited is the last column
patient_full <- patient_full %>%
relocate(recruited, .after = last_col())

# 8. Create sample dataset learning

# A sample of 10% of the full dataset was created to develop initial predictive models with.
# Set seed to create the same sample each time the code is run
set.seed(123)

# Create sample
patient_sample <- slice_sample(patient_full, prop = 0.1)

# 9. Missing Data

# ‘Missing Data’ is when observations are absent for one or more variables and is a common
# occurrence in datasets. When possible, missing data should be minimised. The following
# shows how this has been achieved for missing data in the health and care experience survey.
# When reducing missingness by getting real data is not possible, the approach must be carefully
# considered to avoid introducing bias into the results.

# Create summary of missing data at the variable level using the miss_var function
miss_var_summary <- miss_var_summary(patient_full)

# Create summary of missing data at the case level using the miss_case function
miss_case_summary <- miss_case_summary(patient_full)

# Display missing data summary at the variable level
miss_var_summary

# Display missing data summary at the case level
miss_case_summary

# Remove row number 77749
patient_full <- patient_full[-77749, ]

# 9.1 Visualise missing data

# Rename variables for plotting and pass these to the plot_pattern function
patient_full %>%
rename(c(CHI = prochi, PID = propracticeid, UR8 = ur8fold, OVRd =
overall_decile, INCd = income_decile, EMPd = employment_decile,
HEALTHd = health_decile, EDd = education_decile, ACCd =
access_to_services_decile, HOUd = housing_decile, CRId =
crime_decile, AGEr = age_range, SEX_M = sex_M, SEX_F = sex_F, HB_G
= health_board_glasgow, HB_L = health_board_lanarkshire, HB_T =
health_board_tayside, TIME = route_time, DIST = route_distance,
DDIST = direct_distance, GP = GP_headcount, URC =
urban_rural_category, DEP15 = most_deprived, POP =
practice_population, HACE = percent_positive, REC = recruited)) %>%
select(-c(gpcount_pop_ratio, male_female_ratio, younger_ratio,
middle_ratio, older_ratio, route_speed, date_diff)) %>%
plot_pattern(
rotate = TRUE
)

# Count the number variables containing missing data
patient_full %>%
select(-c(gpcount_pop_ratio, male_female_ratio, younger_ratio,
middle_ratio, older_ratio, route_speed, date_diff)) %>%
{
sum(colSums(is.na(.)) > 0)
}

# 9.2 Explore HACE missing data

#Combining the three HACE datasets and anti-joining with the full cohort of practice IDs in the Cohort dataset shows
#that 10 practices are missing from the HACE datasets that are present in the Cohort dataset.
#These 10 practices had 3674 patients invited to the trial.

# Show number of practices present in Cohort and missing from HACE_clean
anti_join(Cohort, HACE_clean, by = "ProPracticeID") %>%
distinct(ProPracticeID) %>%

# Show number of patients present in Cohort and missing from HACE_clean
anti_join(Cohort, HACE_clean, by = "ProPracticeID") %>%
count()

#An anti-join is performed to compare all HACE practices with all Cohort practices prior to
#matching HACE year and recruitment year. This returns a list of 8, confirming that 2 are
#present in HACE_all but not HACE_clean, demonstrating that 2 practices have a discrepancy
#between Survey_Year and dtFirstContact. This is the reason for them being missing, which
#is confirmed below.

# Show practices present in Cohort_HACE and missing from HACE_all
anti_join(Cohort_HACE, HACE_all, by = "ProPracticeID") %>%
distinct(ProPracticeID, .keep_all = TRUE) %>%
select(ProPracticeID, dtFirstContact) %>%
mutate(ProPracticeID = row_number()) %>%
rename(Practice_Number = ProPracticeID)

#HACE_all dataset is searched for the 10 practice IDs that are missing from HACE_clean since
#this could be due to the relevant survey year being unavailable for some practices. The code is
#omitted since it contains practice identifiers.

#The cause of these two practices missing after filtering matching dtFirstContact and survey
#years is shown below and is due to only having HACE survey 2013/14 available for those
#practices.

#Following contact with Scottish Government, it was not possible to retrieve the correct survey
#years for these two practices. The available survey year was therefore matched for these
#two practices only. The expected effect of this is low, since survey results generally do not
#change significantly between adjacent survey years. The following code is omitted since it
#contains practice IDs. It creates a variable called HACE_new_rows and filters the two relevant
#practices from the Cohort_HACE_GGC dataset. This contains the variables PropracticeID,
#Percentage_Positive and Health_Board_Name.

#Bind rows from HACE_clean and HACE_new_rows to include 
#the additional two practices present in HACE_new_rows
HACE_clean <- bind_rows(HACE_clean, HACE_new_rows)

# Confirm that new dataset contains 156 practices instead of 154
# demonstrating that the two newly matched practices are included
HACE_clean %>%
distinct(ProPracticeID) %>%
count()

#The following omitted code, containing practice IDs, shows the survey year for the 8 missing
#practices to allow provision of these from Scottish Government (SG)

# This shows that the HACE survey year required for the eight missing practices are all 2013/14.
#Investigation by the SG HACE team found that the reason for these missing practice IDs was
#that there was a cluster of practices in the Rutherglen/Cambuslang area that was affected by the
#Lanarkshire and Greater Glasgow and Clyde health board boundary change. This resulted in
#these eight practices having their practice code changed. Therefore, the practices are contained in
#the original HACE dataset, but they have different practice identifiers. The HACE team provided
#the relevant mappings that were placed in the Safe Haven to allow these to be included in the
#linked dataset.

# Import HACE_missing_map
HACE_missing_map <- read_csv("C:/project/missing_map.csv")

# rename variables for matching
HACE_missing_map <- HACE_missing_map %>%
rename(ProPracticeID = "New anoprac", truepropracid = "Old AnoPrac")

# Link missing practices to full HACE dataset
HACE_missing <- HACE_all %>%
left_join(HACE_missing_map, by = "ProPracticeID")

# Change existing ProPracticeIDs to correct IDs while preserving others
HACE_missing <- HACE_missing %>%
mutate(ProPracticeID = coalesce(
truepropracid,
ProPracticeID
))

# Remove observations not from HACE_missing_map
HACE_missing <- HACE_missing %>%
filter(!is.na(truepropracid))

# remove duplicate practice for survey year 13
HACE_missing <- HACE_missing %>%
filter(Survey_Year == 15)

# remove linking variable and survey year
HACE_missing <- HACE_missing %>%
select(-c(truepropracid, Survey_Year))

# confirm 8 practices remain
HACE_missing %>%
distinct(ProPracticeID) %>%
count()

#This provides a dataset of an additional 8 practices with HACE survey data. This is then
#combined with the HACE_clean dataset.
HACE_clean <- bind_rows(HACE_clean, HACE_missing)

#This contains all of the practices present in the original cohort data and removes all missing data
#for the HACE survey. Therefore, there are only 370 (0.005%) rows with missing values in the
#final dataset.

# 9.3 Practice missing data

# The Practice_Dem dataset contains the date that practices registered to take part in the trial and
# help to recruit patient participants.
# check if there are practices present in Practice_Dem and missing from
# Cohort
anti_join(Practice_Dem, Cohort, by = "ProPracticeID") %>%
distinct(ProPracticeID) %>%
count()

# check if there are practices present in Cohort and missing from
# Practice_Dem
anti_join(Cohort, Practice_Dem, by = "ProPracticeID") %>%
distinct(ProPracticeID) %>%
count()

# This shows that 6 practices are present in the Cohort dataset but absent from Practice_Dem.
# This may be because these practices did not recruit any patients.
# check whether recruited = "No" for the 6 practices missing from
# Practice_Dem
anti_join(Cohort, Practice_Dem, by = "ProPracticeID") %>%
distinct(ProPracticeID) %>%
inner_join(patient_full, by = c("ProPracticeID" = "propracticeid")) %>%
filter(recruited == "No") %>%
summarise(
nil_recruitment =
all(recruited == "No")
)

# This confirms that these 6 practices didn’t recruit which explains why they are missing from
# Practice_Dem

# 9.4 Impute missing data

# Imputation of remaining missing data was completed with the mice package using a random
# forest method.
set.seed(123)

# Sample dataset MICE imputation
patient_sample_MICE <- mice(patient_sample, m = 5, defaultMethod = c("rf",
"rf", "rf", "rf"))

# Extract dataset from the imputed output
patient_sample_MICE_data <- complete(patient_sample_MICE)

#Confirm no NAs remain
anyNA(patient_sample_MICE_data)

set.seed(123)
# Full dataset MICE imputation
patient_full_MICE <- mice(patient_full, m = 5, defaultMethod = c("rf",
"rf", "rf", "rf"))

# Extract dataset from the imputed output
patient_full_MICE_data <- complete(patient_full_MICE)
# Confirm no NAs remain in the full imputed dataset
anyNA(patient_full_MICE_data)

# 10. Create test and train samples

# The full and sample datasets were split into training and test samples for the purpose of ML
# modelling.

# 10.1 Create Sample dataset

# Set seed so that the same sample is created each time
set.seed(123)

# Create training dataset containing 80% of the observations
patient_sample_MICE_data_train <- patient_sample_MICE_data %>%
slice_sample(prop = 0.8)

# Create test dataset containing the remaining 20% of the observations
patient_sample_MICE_data_test <- anti_join(patient_sample_MICE_data,
patient_sample_MICE_data_train, by = "prochi")

# Remove identifier variables from training sample
patient_sample_MICE_data_train <- patient_sample_MICE_data_train %>%
select(-c(prochi, propracticeid))

# Remove identifier variables from test sample
patient_sample_MICE_data_test <- patient_sample_MICE_data_test %>%
select(-c(prochi, propracticeid))

# 10.2 Create Full dataset

# Set seed so that the same sample is created each time
set.seed(123)

# Create training dataset containing 80% of the observations
patient_full_MICE_data_train <- patient_full_MICE_data %>%
slice_sample(prop = 0.8)

# Create test dataset containing the remaining 20% of the observations
patient_full_MICE_data_test <- anti_join(patient_full_MICE_data,
patient_full_MICE_data_train, by = "prochi")

# Remove identifier variables from training sample
patient_full_MICE_data_train <- patient_full_MICE_data_train %>%
select(-c(prochi, propracticeid))

# Remove identifier variables from test sample
patient_full_MICE_data_test <- patient_full_MICE_data_test %>%
select(-c(prochi, propracticeid))

# 11. Exploring the Dataset

# The summary function provides a high-level overview of the complete patient_full dataset.
# Summarise dataset

summary(patient_full)

# Number and percentage of recruited patients of those invited
patient_full %>%
select(recruited) %>%
group_by(recruited) %>%
summarise(count = n()) %>%
mutate(percentage = (count / sum(count)) * 100) %>%
arrange(desc(recruited))

# Plot bar chart of proportion of recruited patients invited
ggplot(patient_full, aes(x = recruited, y = after_stat(prop), group = 1))
  +
geom_bar()

# Display the number and percentage of patients recruited per practice in
# descending order
patient_full %>%
select(propracticeid, recruited) %>%
group_by(propracticeid) %>%
summarise(
count_yes = sum(recruited == "Yes"),
count_no = sum(recruited == "No"),
recruited_percent = round((count_yes / (count_yes + count_no)) * 100,
1)
) %>%
mutate(propracticeid = row_number()) %>%
rename(practice_name = propracticeid) %>%
arrange(desc(count_yes)) %>%
mutate(across(everything(), ~
case_when(.x < 5 ~ "<5",
TRUE ~ as.character(.x)))
)

# Display summary statistics for the number of patients recruited per
# practice
patient_full %>%
select(propracticeid, recruited) %>%
group_by(propracticeid) %>%
summarise(
count_yes = sum(recruited == "Yes"),
count_no = sum(recruited == "No")
) %>%
mutate(propracticeid = row_number()) %>%
summarise(Min = min(count_yes), Max = max(count_yes), Mean =
mean(count_yes), Median = median(count_yes), IQR = IQR(count_yes),
SD = sd(count_yes))

patient_full %>%
summarise(count = n_distinct(propracticeid))

# 12. Machine Learning

# 12.1 Sample models

# trainControl This provides a standard set of instructions that are common to all ML methods
# used. The summaryFunction = twoClassSummary determines that AUC rather than accuracy
# should be used to tune the parameters. This function as defined, computes the sensitivity and
# specificity with a 50% probability cutoff, the area under the ROC curve a 2-class prediction
# problem, and to use 10-fold cross-validation.

# define trainControl object
trControl <- trainControl(
method = "cv",
number = 10,
summaryFunction = twoClassSummary,
classProbs = TRUE,
verboseIter = TRUE,
savePredictions = TRUE,
)

#Training on sample of data 
#The following trains machine learning models on sample data.

# train ranger model on sample data
model_sample_ranger_00 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "ranger",
preProcess = c("center", "scale"),
trControl = trControl
)

# Display results of ranger model
print(model_sample_ranger_00, showSD = TRUE)

# Display maximum AUC for ranger model
print(sprintf("maximum AUC: %.3f",
max(model_sample_ranger_00[["results"]][["ROC"]])))

# Train glmnet model on sample data
model_sample_glmnet_00 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "glmnet",
preProcess = c("zv", "center", "scale", "pca"),
trControl = trControl
)

# Display results of glmnet model
print(model_sample_glmnet_00, showSD = TRUE)

# Display maximum AUC for glmnet model
print(sprintf("maximum AUC: %.3f",
max(model_sample_glmnet_00[["results"]][["ROC"]])))

# glmnet custom model 
# The following defines specific alpha and lamda values as well as
# additional pre-processing -center, scale and principle components analysis – since glmnet models
# generally benefit from this.
# specify range of alpha and lamda values to train on
my_grid <- expand.grid(
alpha = 0:0.5:1,
lambda = seq(0.0001, 1, length = 20)
)

# Train custom glmnet model on sample data
model_sample_glmnet_01 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
tuneGrid = my_grid,
method = "glmnet",
preProcess = c("zv", "center", "scale", "pca"),
trControl = trControl
)

# Display results of custom glmnet model
print(model_sample_glmnet_01, showSD = TRUE)

# Display maximum AUC for custom glmnet model
print(sprintf("maximum AUC: %.3f",
max(model_sample_glmnet_01[["results"]][["ROC"]])))

# Train xgbtree model on sample data
model_sample_xgbtree_00 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "xgbTree",
trControl = trControl
)

# Display results of xgbtree model
print(model_sample_xgbtree_00, showSD = TRUE)

# Display maximum AUC for xgbtree model
print(sprintf("maximum AUC: %.3f",
max(model_sample_xgbtree_00[["results"]][["ROC"]])))

# Train neural network model on sample data
model_sample_nnet_00 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "nnet",
trControl = trControl
)

# Display results of neural network model
print(model_sample_nnet_00, showSD = TRUE)

print(sprintf("maximum AUC: %.3f",
max(model_sample_nnet_00[["results"]][["ROC"]])))

# Train KNN model on sample data
model_sample_knn_00 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "kknn",
trControl = trControl
)

# Display results of KNN model
print(model_sample_knn_00, showSD = TRUE)

# Display maximum AUC for KNN model
print(sprintf("maximum AUC: %.3f",
max(model_sample_knn_00[["results"]][["ROC"]])))

# Train Naïve Bayes model on sample data
model_sample_nb_00 <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "naive_bayes",
preProcess = c("zv", "center", "scale", "pca"),
trControl = trControl
)

# Display results of Naïve Bayes model
print(model_sample_nb_00, showSD = TRUE)

# Display maximum AUC for Naïve Bayes model
print(sprintf("maximum AUC: %.3f",
max(model_sample_nb_00[["results"]][["ROC"]])))

# 12.2 Comparing 1st generation models

# create a list of sample models
sample_model_list <- list(ranger00 = model_sample_ranger_00, glmnet00 =
model_sample_glmnet_00, glmnet01 = model_sample_glmnet_01, xgbtree =
model_sample_xgbtree_00, nnet = model_sample_nnet_00, knn =
model_sample_knn_00, NB = model_sample_nb_00)

# collate resampling results from sample list
resamples <- resamples(sample_model_list)

# provide summary of the results for all models
summary(resamples)

# Create plot of AUC for all models
dotplot(resamples, metric = "ROC")

# 12.3 Optimising xgboost parameters

# define traincontrol for adaptive resampling
trControl_adaptive <- trainControl(
method = "adaptive_cv",
number = 10,
repeats = 10,
adaptive = list(
min = 5,
alpha = 0.05,
method = "gls",
complete = TRUE
),
search = "random",
summaryFunction = twoClassSummary,
classProbs = TRUE,
verboseIter = TRUE,
savePredictions = TRUE
)

# train model with adaptive resampling
model_sample_xgbtree_adaptive <- train(recruited ~ .,
data = patient_sample_MICE_data_train,
method = "xgbTree",
trControl = trControl_adaptive,
tuneLength = 100
)

# 12.4 Training on complete dataset

# xgboost is used to train on the full dataset since it was the best performing model on the sample
# data. Initially the parameters identified by adaptive resampling are used. This is then compared
# to a default xgboost model.

# Specify tuneGrid using optimal parameters from adaptive
# resampling of sample
xgb_grid <- expand.grid(
nrounds = 288,
max_depth = 1,
eta = 0.555,
gamma = 1.82,
colsample_bytree = 0.316,
min_child_weight = 13,
subsample = 0.7
)

#Default xgboost model

#Train model without specifying parameters
model_full_xgbtree_00 <- train(recruited ~ .,
data = patient_full_MICE_data_train,
method = "xgbTree",
preProcess = c("center", "scale"),
trControl = trControl
)

# Display results of default xgboost model
print(model_full_xgbtree_00, showSD = TRUE)

# Display maximum AUC for default xgboost model
print(sprintf("maximum AUC: %.3f",
max(model_full_xgbtree_00[["results"]][["ROC"]])))

#Defined parameter xgboost model 
#The following model used the parameters identified by
#adaptive resampling.
# train xgbtree using parameters from adaptive resampling contained
#within xgb_grid
model_full_xgbtree_10 <- train(recruited ~ .,
data = patient_full_MICE_data_train,
method = "xgbTree",
tuneGrid = xgb_grid,
preProcess = c("center", "scale"),
trControl = trControl
)

# Display results of adaptive resampling model
print(model_full_xgbtree_10, showSD = TRUE)

# Display maximum AUC for adaptive resampling model
print(sprintf("maximum AUC: %.3f",
max(model_full_xgbtree_10[["results"]][["ROC"]])))

# 12.5 Compare final models

# Create list containing both xgboost models trained on full dataset
model_full_list <- list(xgbtree00 = model_full_xgbtree_00, xgbtree10 =
model_full_xgbtree_10)

# collate resampling results from full list
resamples <- resamples(model_full_list)

# provide summary of the results for all models
summary(resamples)

# Create plot of AUC for both xgbtree models
dotplot(resamples, metric = "ROC")

# 12.6 Variable Importance

# xgbtree variable importance
varImp(model_full_xgbtree_00)

# 12.7 Plot of final model

plot(model_full_xgbtree_00)

# 12.8 Predicting test data

# After models were optimised on training data, the test dataset sample was used for testing models.
# This allows for internal validation of the model using an unseen set of data.

# 12.8.1 Confusion Matrix

# use predict function to predict outcomes of test data using xgbtree00
# model
predict_model_xgbtree_00 <- predict(model_full_xgbtree_00,
patient_full_MICE_data_test, type = "prob")

# select the "No" column from the predicted outcomes
predict_model_xgbtree_00 <- predict_model_xgbtree_00 %>%
select(No)

# change probability threshold to 0.84 for making a "No" prediction
p_class_xgbtree_00 <- ifelse(predict_model_xgbtree_00 > 0.84, "No",
"Yes")

# Convert to factor variable
p_class_xgbtree_00 <- as.factor(p_class_xgbtree_00)

#Create confusion matrix
confusionMatrix(p_class_xgbtree_00,
patient_full_MICE_data_test$recruited)

# 12.8.2 Receiver Operator Characteristic curves

# The pROC package is used to create an ROC curve
# to overlay test and train AUCs for the best performing xgboost model.

# Create vector of predicted recruited values for test dataset
pROC_predicted_xgbtree_00_test <- predict(model_full_xgbtree_00,
patient_full_MICE_data_test, type = "prob") %>%
select(Yes)%>%
rename(recruited = Yes) %>%
pull()

# Create vector of test dataset response values, recoding values to allow
# comparison with predictions
pROC_response_test <- patient_full_MICE_data_test%>%
mutate(recruited = recode(recruited, "No" = 0, "Yes" = 1)) %>%
pull(recruited)

# Create vector of predicted recruited values for train dataset
pROC_predicted_xgb_tree_00_train <- predict(model_full_xgbtree_00,
patient_full_MICE_data_train, type = "prob") %>%
select(Yes) %>%
rename(recruited = Yes) %>%
pull()

# Create vector of train dataset response dataset, recoding values to
# allow comparison with predictions
pROC_response_train <- patient_full_MICE_data_train %>%
mutate(recruited = recode(recruited, "No" = 0, "Yes" = 1)) %>%
pull(recruited)

# plot ROC curves for test and train predictions
roc_xgbtree_00_test <- roc(pROC_response_test,
pROC_predicted_xgbtree_00_test, ci = TRUE)
roc_xgbtree_00_train <- roc(pROC_response_train,
pROC_predicted_xgb_tree_00_train, ci = TRUE)

#auc with 95% CI for test dataset
roc_xgbtree_00_test

#auc with 95% CI for training dataset
roc_xgbtree_00_train

#Create list of train and test ROCs to pass to ggroc function
roc.list <- list(test = roc_xgbtree_00_test, train =
roc_xgbtree_00_train)

#Overlay ROC for test and train datasets
ggroc(roc.list)