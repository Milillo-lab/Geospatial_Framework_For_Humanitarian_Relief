# Urban Damage and Infrastructure Analysis Pipeline
## Demo for Kahramanmaraş City

The user is invited to try out different configurations, options etc. for the workflow.
This Demo provided here is just a representation of the capabilities of the code
for the Kahramanmaraş - Türkiye 2023 Earthquakes. Still, it is highly recommended to read through
the main README.md file instructions in main GitHub repository.

# Zenodo - Data Acqusition
https://zenodo.org/records/17851494?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImViZGMwYmNmLTgyYmItNGI4Yy04MWM3LTg5ZDNhNjllZTM1MiIsImRhdGEiOnt9LCJyYW5kb20iOiJmZmU1YWQ0MzQ0NDMwNDYzNmJjMzdkMTU5OTVlODBiMiJ9.SQEScrM6akmSsAiQfr4ZQqJ6YoSsDrpkpivlbwByWlN8W0ZtE6Ew3Xz2U-NrqHy_6MmsvQ58pKsB_Rz5fd_pSA
User is invited to download the necessary files from this link.

# Important
 As stated in the main instructions, user should put the two necessary Data
	1. DPM (https://zenodo.org/records/15369579)
	2. Population (Zenodo link above)
		into the ./Data folder. 
 How_to_Downloads - instructions can be followed if necessary.
 Also, in the case of Building downloading (Section A) is not working due to the server issues, 	
cleaned_buildings_selected_region.rar can be downloaded from the Zenodo and its content pasted inside ./Outputs/A - This will allow the HADAW to work without problems in the incoming sections.
	
## The code is modular;
 once completely run, all of the pipeline can get the necessary data from
./Outputs/A-I folders to run itself. Hence if for instance;
Building Data Download Server is not working -> User is invited to download said data manually and paste it into ./Outputs/A folder. This will allow the code to read it from there and work without a problem.
