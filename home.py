from io import StringIO
import streamlit as st
import pandas as pd
import json, re, csv, asyncio
from utils.Ollama_Agent import extract_Ollama
from utils.OpenAI_Agent import extract_OpenAI

SYSTEM_PROMPT_PARAGRAPH ='''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goals:
    - extract the stylistic and content changes made by the attending physician to the resident report
    - present the changes ***as succinctly as possible*** while still being readable
    - treat this as feedback meant to inform the resident on how they can improve their report writing to better match the attending
    - The format of the output should include the following headings: Findings, Impression, Stylistic Approach, Change Characteristics
    - Regarding Change Characteristics, the following rating system will be used. Please identify ***all*** instances of each type and provide a brief explanation for each number chosen (multiple numbers can be used, and repeat numbers can be used for different examples within the reports)
        * 1: Addition of missing positive findings (e.g. "Lung Bases: Unremarkable" --> "Scattered subcentimeter nodules likely incidental.")
        * 2: Deletion of (incorrect) positive findings (e.g. “Small left pleural effusion is noted, possibly related to recent infection.” --> “No definite pleural effusion is identified; left basilar opacity likely reflects adjacent atelectasis.”)
        * 3: Addition of negative findings (e.g. “Lungs are clear with no consolidation.” --> “Lungs are clear with no consolidation, effusion, or pneumothorax.”)
        * 4: Correction of the expression of findings / Proofreading (e.g. “Pancreas has hazy borders suggestive of inflammation.” --> “The pancreas demonstrates ill-defined margins with surrounding stranding, consistent with pancreatitis.”)
        * 5: Correction of the diagnosis (e.g. “Thickened bowel loops likely represent Crohn’s disease.” --> “Thickened distal ileum may represent infectious or inflammatory ileitis; Crohn’s is a consideration but not definitive.”)
        * 6: Follow-up exam or treatment recommendations (e.g. “Stable hepatic lesion, likely benign hemangioma.” --> “Stable hepatic lesion measuring 1.5 cm, likely benign hemangioma. Recommend 6-month follow-up MRI to confirm stability.”)
        * 7: Level of certainty of finding (e.g. “There is a 4 mm right upper lobe nodule that could represent malignancy.” --> “4 mm right upper lobe nodule is indeterminate, but likely benign given size and morphology.”)
    - Under **Change Characteristics**, return bullet points in the following format (this format must be strictly followed):

        - [number]: "[Resident quote]" --> "[Attending quote]"

            [One-sentence explanation of the change]
    - Under each heading will be a ***short summary paragraph*** of the improvements that could be made regarding the given headings – an example is shown below

Example:
"**Findings:**

The attending version removes some descriptive qualifiers and incidental findings, focusing on clinically relevant features while standardizing language. Specific changes include simplifying liver lesion descriptions, omitting details about the appendix, ascites, and cystic lesion characteristics that the resident included. The attending adds minor incidental lung nodules that were not mentioned in the resident's version.

**Impression:**

Both reports convey the same major findings but the attending condenses phrasing, focusing on diagnostic clarity without repeating measurement values unnecessarily.

**Stylistic Approach:**

The attending emphasizes brevity and clarity, using terms like "normal" instead of "unremarkable," omitting redundant or non-actionable details, and ensuring a standardized structure that’s easier to scan for key findings.

**Change Characteristics**
- 1: “Unremarkable.” --> “No focal consolidation. Scattered subcentimeter nodules likely incidental.”

    The attending included specific incidental findings to add clinical nuance.

- 4: "The liver demonstrates a heterogeneous lesion in the right hepatic lobe measuring 4.2 x 3.8 cm, consistent with a hepatic adenoma. This lesion was smaller on prior imaging, previously measuring 2.5 x 2.0 cm, indicating interval growth. There is associated mild contour nodularity suggesting early chronic changes." --> "Interval enlargement of a right hepatic lobe lesion, now measuring 4.2 x 3.8 cm, consistent with a hepatic adenoma. Mild surface nodularity is noted."

    The language was streamlined to emphasize key changes while reducing redundancy.

- 4: "The pancreas demonstrates an ill-defined hypodense area involving the pancreatic tail measuring approximately 3.7 x 2.9 cm, consistent with complex pancreatitis. There is surrounding inflammatory stranding and an adjacent phlegmon measuring approximately 4.5 x 3.1 cm. No discrete fluid collection is identified." --> "Complex inflammatory changes and ill-defined low-attenuation area in the pancreatic tail measuring 3.7 x 2.9 cm. Adjacent phlegmon measuring 4.5 x 3.1 cm."

    The phrasing was tightened for clarity and radiologic convention.

- 4: "The uterus is enlarged with multiple fibroids, the largest located in the anterior fundal region measuring 5.4 x 4.8 cm. There is a left adnexal cystic lesion measuring 4.2 x 3.9 x 4.0 cm, likely representing an ovarian cyst. No solid components or septations are identified." --> "Normal."

    The attending omitted detail, favoring a high-level summary likely reflecting clinical priorities.
"
'''

SYSTEM_PROMPT_TABLE ='''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goals:
    - extract the stylistic and content changes made by the attending physician to the resident report
    - present the changes ***as succinctly as possible*** while still being readable
    - treat this as feedback meant to inform the resident on how they can improve their report writing to better match the attending
    - The format of the output should be a CSV table with the following columns: Section, Resident Report, Attending Report, Difference Type
    - The rows should ***match the sections of the findings***
    - Regarding the "Difference Type" Column, it will be a ***list of numbers (i.e. allowed repeats for multiple identifications, multiple numbers allowed in list)*** for the following categories:
        * 1: Addition of missing positive findings (e.g. "Lung Bases: Unremarkable" --> "Scattered subcentimeter nodules likely incidental.")
        * 2: Deletion of (incorrect) positive findings (e.g. “Small left pleural effusion is noted, possibly related to recent infection.” --> “No definite pleural effusion is identified; left basilar opacity likely reflects adjacent atelectasis.”)
        * 3: Addition of negative findings (e.g. “Lungs are clear with no consolidation.” --> “Lungs are clear with no consolidation, effusion, or pneumothorax.”)
        * 4: Correction of the expression of findings / Proofreading (e.g. “Pancreas has hazy borders suggestive of inflammation.” --> “The pancreas demonstrates ill-defined margins with surrounding stranding, consistent with pancreatitis.”)
        * 5: Correction of the diagnosis (e.g. “Thickened bowel loops likely represent Crohn’s disease.” --> “Thickened distal ileum may represent infectious or inflammatory ileitis; Crohn’s is a consideration but not definitive.”)
        * 6: Follow-up exam or treatment recommendations (e.g. “Stable hepatic lesion, likely benign hemangioma.” --> “Stable hepatic lesion measuring 1.5 cm, likely benign hemangioma. Recommend 6-month follow-up MRI to confirm stability.”)
        * 7: Level of certainty of finding (e.g. “There is a 4 mm right upper lobe nodule that could represent malignancy.” --> “4 mm right upper lobe nodule is indeterminate, but likely benign given size and morphology.”)
    - Return only raw CSV (***NO EXPLANATION, MARKDOWN, OR EXTRA TEXT ASIDE FROM THE CSV***), and include headers in the first row

Example:
"Section","Resident Report","Attending Report","Difference Type"
"Lung Bases","Unremarkable.","No focal consolidation. Scattered subcentimeter nodules.","1"
"Liver","Heterogeneous lesion", "mild contour nodularity.","Interval enlargement, mild surface nodularity.","4"
"Biliary System","Explicit duct sizes, no obstructing mass mentioned.","Duct sizes, no mention of obstruction.",""
"Pancreas","Complex pancreatitis, no discrete fluid collection.","Complex changes, omits fluid collection comment.","4"
"Spleen/Adrenals/Kidneys","Unremarkable.","Normal.",""
"Pelvis/Bladder","Ovarian cyst description includes "no solid components."","No mention of solid components.","4"
"Bowel","Appendix unremarkable.","Appendix not mentioned.",""
"Mesentery/Peritoneum","Notes no ascites.","Ascites not mentioned.",""
"Bones & Soft Tissues","Degenerative changes, no aggressive lesion.","Degenerative changes only.",""
"Style Overall","Detailed, more explanatory.","Concise, standardized, focused on clinical impact.",""
'''

SYSTEM_PROMPT_PARAGRAPH_MULTI ='''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goals:
    - extract the stylistic and content changes made by the attending physician to the resident report
    - present the changes ***as succinctly as possible*** while still being readable
    - treat this as feedback meant to inform the resident on how they can improve their report writing to better match the attending
    - The format of the output should include the following headings: Findings, Impression, Stylistic Approach
    - Under each heading will be a ***short summary paragraph*** of the improvements that could be made regarding the given headings – an example is shown below

Example:
"**Findings:**

The attending version removes some descriptive qualifiers and incidental findings, focusing on clinically relevant features while standardizing language. Specific changes include simplifying liver lesion descriptions, omitting details about the appendix, ascites, and cystic lesion characteristics that the resident included. The attending adds minor incidental lung nodules that were not mentioned in the resident's version.

**Impression:**

Both reports convey the same major findings but the attending condenses phrasing, focusing on diagnostic clarity without repeating measurement values unnecessarily.

**Stylistic Approach:**

The attending emphasizes brevity and clarity, using terms like "normal" instead of "unremarkable," omitting redundant or non-actionable details, and ensuring a standardized structure that’s easier to scan for key findings.
"
'''

AGENT_PROMPT_1 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy the addition of positive findings – when the attending report contains a positive finding that the resident report lacks
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Resident Report", "Attending Report", "Difference Type", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "1" if a positive finding is identified for a given section
            - Explanation: Consice 1-sentence summary explaining the observed addition of positive finding, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with a positive finding

Inclusion Criteria:
    - Addition of positive finding
    - Addition of detail that contributes to positive finding
Exclusion Criteria:
    - Addition of negative finding
    - Phrasing adjustments/correction of expression used
    - Correction of a diagnosis
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
    - Adjusting the level of certainty

Example Inclusion:

Input:
    Resident Report: "Lung Bases: Unremarkable."
    Attending Report: "Lung Bases: No focal consolidation. Scattered subcentimeter nodules likely incidental."

Output:
    Section: Lung Bases
    Resident Report: Unremarkable.
    Attending Report: No focal consolidation. Scattered subcentimeter nodules likely incidental.
    Difference Type: 1
    Explanation: The attending added previously unmentioned subcentimeter lung nodules, highlighting clinically nuanced findings absent from the resident’s draft.

Example Exclusion:

Input:
    Resident Quotation: "No ascites."
    Attending Quotation: "No ascites. No abnormal focal fluid collections. No free air."
Explanation: While an addition is made, it is an addition of ***negative*** findings, and should not be included

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Lung Bases","Unremarkable.","No focal consolidation. Scattered subcentimeter nodules likely incidental.","1","The attending added previously unmentioned subcentimeter lung nodules, highlighting clinically nuanced findings absent from the resident’s draft."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"

Example Output 3:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Spleen/Adrenals/Kidneys","Unremarkable.","Incidental 3.5 x 3.2 cm mass in the upper pole of the right kidney.","1","The attending added previously unmentioned mass in the right kidney."
'''

AGENT_PROMPT_2 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy the deletion of ***incorrect*** positive findings – when the attending report removes a finding that the resident report contained
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Difference Type", "Quote Identified", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "2" if a positive finding is removed for a given section
            - Explanation: Consice 1-sentence summary explaining the observed removal of positive finding, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with a deletion of a positive finding

Inclusion Criteria:
    - Deletion of positive finding
Exclusion Criteria:
    - Addition of positive finding
    - Addition of negative finding
    - Exclusion of details that contribute to a positive finding (as long as the main finding is present in both resident and attending reports)
    - Phrasing adjustments/correction of expression used
    - Correction of a diagnosis
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
    - Adjusting the level of certainty

Example Inclusion:

Input:
    Resident Report: "Mesentery/Peritoneum: Mild inflammatory stranding is present surrounding the aforementioned right lower quadrant collection. No free air is identified outside this loculated process. No additional abnormal fluid collections are seen."
    Attending Report: "Mesentery/Peritoneum: No free air."

Output:
    Section: Mesentary/Peritoneum
    Resident Report: Mild inflammatory stranding is present surrounding the aforementioned right lower quadrant collection. No free air is identified outside this loculated process. No additional abnormal fluid collections are seen.
    Attending Report: No free air.
    Difference Type: 2
    Explanation: "The attending did not identify surrounding inflammatory stranding."

Example Exclusion:

Input:
    Resident Quotation: "Multiple gallstones present within the gallbladder, consistent with cholelithiasis. There is no gallbladder wall thickening or pericholecystic fluid to suggest acute cholecystitis. Compared to the prior study, there is increased prominence of pericholecystic fat stranding, suggesting interval worsening of gallbladder pathology, although no acute inflammatory signs are present."
    Attending Quotation: "Cholelithiasis with increased pericholecystic fat stranding compared to prior."
Explanation: While the attending quotation omits information, all major findings are present, and the overall information communicated is the same.

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Lung Bases","Mild inflammatory stranding is present surrounding the aforementioned right lower quadrant collection. No free air is identified outside this loculated process. No additional abnormal fluid collections are seen.","No free air.","2","The attending did not identify surrounding inflammatory stranding."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
'''

AGENT_PROMPT_3 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy the addition of negative findings – when the attending report contains a negative finding that the resident report contained
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Difference Type", "Quote Identified", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "3" if a negative finding is identified for a given section
            - Explanation: Consice 1-sentence summary explaining the observed addition of negative finding, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with an addition of a negative finding finding

Inclusion Criteria:
    - Addition of negative finding
Exclusion Criteria:
    - Addition of positive finding
    - Addition of details that contribute to a positive finding
    - Phrasing adjustments/correction of expression used
    - Correction of a diagnosis
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
    - Adjusting the level of certainty

Example Inclusion:

Input:
    Resident Report: "Mesentary/Peritoneum: No ascites."
    Attending Report: "Mesentary/Peritoneum: No ascites. No abnormal focal fluid collections. No free air."

Output:
    Section: Mesentary/Peritoneum
    Resident Report: No ascites.
    Attending Report: No ascites. No abnormal focal fluid collections. No free air.
    Difference Type: 3
    Explanation: "Important negative findings (fluid collections, free air) added for completeness."

Example Exclusion:

Input:
    Resident Quotation: "Lung Bases: Unremarkable."
    Attending Quotation: "Lung Bases: No focal consolidation. Scattered subcentimeter nodules likely incidental."
Explanation: While the attending quotation adds new information, it is a positive finding (even if only incidental).

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Mesentary/Peritoneum","No ascites.","No ascites. No abnormal focal fluid collections. No free air.","3","Important negative findings (fluid collections, free air) added for completeness."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
'''

AGENT_PROMPT_4 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy the correction of the expression of findings – when the attending report contains a ***major*** rephrasing of resident findings (general information communicated is the same, but largely different phrasing)
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Difference Type", "Quote Identified", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "4" if a correction of expression is made
            - Explanation: Consice 1-sentence summary explaining the correction/rephrasing of resident findings, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with major corrections of expression/rephrasing

Inclusion Criteria:
    - ***Major*** phrasing adjustments/correction of expression used
Exclusion Criteria:
    - ***Minor*** phrasing adjustments/correction of expression used (e.g. "Normal" --> "Unremarkable")
    - Addition of positive finding
    - Addition of details that contribute to a positive finding
    - Correction of a diagnosis
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
    - Adjusting the level of certainty

Example Inclusion:

Input:
    Resident Quotation: "Pancreas: The pancreas demonstrates an ill-defined hypodense area involving the pancreatic tail measuring approximately 3.7 x 2.9 cm, consistent with complex pancreatitis. There is surrounding inflammatory stranding and an adjacent phlegmon measuring approximately 4.5 x 3.1 cm. No discrete fluid collection is identified."
    Attending Quotation: "Pancreas: Complex inflammatory changes and ill-defined low-attenuation area in the pancreatic tail measuring 3.7 x 2.9 cm. Adjacent phlegmon measuring 4.5 x 3.1 cm."

Output:
    Section: Pancreas
    Resident Report: The pancreas demonstrates an ill-defined hypodense area involving the pancreatic tail measuring approximately 3.7 x 2.9 cm, consistent with complex pancreatitis. There is surrounding inflammatory stranding and an adjacent phlegmon measuring approximately 4.5 x 3.1 cm. No discrete fluid collection is identified.
    Attending Report: Complex inflammatory changes and ill-defined low-attenuation area in the pancreatic tail measuring 3.7 x 2.9 cm. Adjacent phlegmon measuring 4.5 x 3.1 cm.
    Difference Type: 4
    Explanation: "The attending streamlines and clarifies the description, condensing phrasing for improved clarity and radiologic convention."

Example Exclusion:

Input:
    Resident Quotation: "Spleen: Unremarkable."
    Attending Quotation: "Spleen: Normal."
Explanation: While the attending quotation uses different phrasing than the resident, it is a minor change that should not be included.

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Pancreas","The pancreas demonstrates an ill-defined hypodense area involving the pancreatic tail measuring approximately 3.7 x 2.9 cm, consistent with complex pancreatitis. There is surrounding inflammatory stranding and an adjacent phlegmon measuring approximately 4.5 x 3.1 cm. No discrete fluid collection is identified.","Complex inflammatory changes and ill-defined low-attenuation area in the pancreatic tail measuring 3.7 x 2.9 cm. Adjacent phlegmon measuring 4.5 x 3.1 cm.","3","The attending streamlines and clarifies the description, condensing phrasing for improved clarity and radiologic convention."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
'''

AGENT_PROMPT_5 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy the correction of a diagnosis – when the attending report contains a correction of the diagnosis from the resident's report
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Difference Type", "Quote Identified", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "5" if a correction of a diagnosis is made
            - Explanation: Consice 1-sentence summary explaining the correction of the diagnosis, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with major corrections of a diagnosis

Inclusion Criteria:
    - Correction of diagnosis
Exclusion Criteria:
    - Deletion of positive findings
    - Deletion of negative findings
    - Addition of positive finding
    - Addition of details that contribute to a positive finding
    - Addition of negative finding
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
    - Adjusting the level of certainty

Example Inclusion:

Input:
    Resident Report: "Bowel: Thickened bowel loops likely represent Crohn’s disease."
    Attending Report: "Bowel: Thickened distal ileum may represent infectious or inflammatory ileitis."

Output:
    Section: Pancreas
    Resident Report: Thickened bowel loops likely represent Crohn’s disease.
    Attending Report: Thickened distal ileum may represent infectious or inflammatory ileitis.
    Difference Type: 5
    Explanation: "The attending identifies a different diagnosis than the resident."

Example Exclusion:

Input:
    Resident Quotation: "Lung Bases: Unremarkable."
    Attending Quotation: "Lung Bases: No focal consolidation. Scattered subcentimeter nodules likely incidental."
Explanation: While the attending quotation includes new information, it is not indicative of a different diagnosis, and should not be included.

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Pancreas","Thickened bowel loops likely represent Crohn’s disease.","Thickened distal ileum may represent infectious or inflammatory ileitis.","5","The attending identifies a different diagnosis than the resident."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
'''

AGENT_PROMPT_6 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy the addition of a follow-up exam or treatment plan – when the attending report contains a follow-up exam or treatment plan that is not present in the resident's report
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Difference Type", "Quote Identified", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "6" if a follow-up exam or treatment plan is added
            - Explanation: Consice 1-sentence summary explaining the addition of a follow-up exam or treatment plan, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with major corrections of a diagnosis

Inclusion Criteria:
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
Exclusion Criteria:
    - Deletion of positive findings
    - Deletion of negative findings
    - Addition of positive finding
    - Addition of details that contribute to a positive finding
    - Addition of negative finding
    - Adjusting the level of certainty
    - Correction of diagnosis

Example Inclusion:

Input:
    Resident Quotation: "IMPRESSION: 1. Ruptured appendicitis with a periappendiceal abscess measuring approximately 5.7 x 4.9 x 6.2 cm, containing gas locules and surrounded by inflammatory changes.\n2. Cholelithiasis without definitive evidence of acute cholecystitis; however, interval worsening of pericholecystic fat stranding compared to prior study suggests progression of gallbladder disease.\n3. Newly identified cystic lesion in the pancreatic head measuring 2.8 x 2.3 cm, likely representing a pancreatic cyst.\n4. Mild contour irregularity and periportal edema in the liver, not previously noted, concerning for early chronic liver changes."
    Attending Quotation: "IMPRESSION: 1. Ruptured appendicitis with abscess formation.\n2. Incidental right renal mass requiring further evaluation.\n3. Pancreatic head cystic lesion.\n4. Mild new periportal edema."

Output:
    Section: IMPRESSION
    Resident Report: 1. Ruptured appendicitis with a periappendiceal abscess measuring approximately 5.7 x 4.9 x 6.2 cm, containing gas locules and surrounded by inflammatory changes.\n2. Cholelithiasis without definitive evidence of acute cholecystitis; however, interval worsening of pericholecystic fat stranding compared to prior study suggests progression of gallbladder disease.\n3. Newly identified cystic lesion in the pancreatic head measuring 2.8 x 2.3 cm, likely representing a pancreatic cyst.\n4. Mild contour irregularity and periportal edema in the liver, not previously noted, concerning for early chronic liver changes.
    Attending Report: 1. Ruptured appendicitis with abscess formation.\n2. Incidental right renal mass requiring further evaluation.\n3. Pancreatic head cystic lesion.\n4. Mild new periportal edema.
    Difference Type: 6
    Explanation: "The attending gives a recommendation for follow-up imaging for a renal mass."

Example Exclusion:

Input:
    Resident Quotation: "Lung Bases: Unremarkable."
    Attending Quotation: "Lung Bases: No focal consolidation. Scattered subcentimeter nodules likely incidental."
Explanation: While the attending quotation includes new information, it is not indicative of a follow up study or treatment plan.

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Pancreas","1. Ruptured appendicitis with a periappendiceal abscess measuring approximately 5.7 x 4.9 x 6.2 cm, containing gas locules and surrounded by inflammatory changes.\n2. Cholelithiasis without definitive evidence of acute cholecystitis; however, interval worsening of pericholecystic fat stranding compared to prior study suggests progression of gallbladder disease.\n3. Newly identified cystic lesion in the pancreatic head measuring 2.8 x 2.3 cm, likely representing a pancreatic cyst.\n4. Mild contour irregularity and periportal edema in the liver, not previously noted, concerning for early chronic liver changes.","1. Ruptured appendicitis with abscess formation.\n2. Incidental right renal mass requiring further evaluation.\n3. Pancreatic head cystic lesion.\n4. Mild new periportal edema.","6","The attending gives a recommendation for follow-up imaging for a renal mass."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
'''

AGENT_PROMPT_7 = '''You are a feedback tool that compares radiologist resident report drafts with the final attending physician report.

Goal: Identitfy an adjustment in the level of certainty within the report – when the attending report adjusts the confidence of a finding that is present in the resident's report
    - This will be done on a section-by-section basis (e.g. compare the "liver" section of resident and attending reports)
    - Please adhere ***strictly*** to the below inclusion and exclusion criteria. Do NOT include any findings if it does not fit into the below criteria
    - Output format: csv format as described below – DO NOT INCLUDE EXPLANATIONS OR TRAILING MARKS/WRAPPERS
        - Columns: "Section", "Difference Type", "Quote Identified", "Explanation"
            - Section: The name of the section (e.g. "Lung Bases", "Liver", "Biliary System", etc.)
            - Resident Report: The section text of the ***resident report***
            - Attending Report: The section text of the ***attending report***
            - Difference Type: The number "7" if a confidence adjustment is made
            - Explanation: Consice 1-sentence summary explaining the adjusted report confidence, phrased as feedback for the resident
        - Rows: The rows should ***match the sections of the findings*** but ONLY include sections with major corrections of a diagnosis

Inclusion Criteria:
    - Adjusting the level of certainty
Exclusion Criteria:
    - Deletion of positive findings
    - Deletion of negative findings
    - Addition of positive finding
    - Addition of details that contribute to a positive finding
    - Addition of negative finding
    - Addition of a follow-up exam
    - Addition of a treatment recommendation
    - Adjusting the level of certainty
    - Correction of diagnosis

Example Inclusion:

Input:
    Resident Quotation: "Bowel: There is a mass in the right lower quadrant, likely representing an appendiceal abscess."
    Attending Quotation: "Bowel: There is a soft tissue density in the right lower quadrant, which may represent an appendiceal abscess; correlation with clinical findings is recommended."

Output:
    Section: Bowel
    Resident Report: There is a mass in the right lower quadrant, likely representing an appendiceal abscess.
    Attending Report: There is a soft tissue density in the right lower quadrant, which may represent an appendiceal abscess; correlation with clinical findings is recommended.
    Difference Type: 7
    Explanation: "The attending softened the diagnostic certainty and emphasized the need for clinical correlation."

Example Exclusion:

Input:
    Resident Quotation: "Lung Bases: Unremarkable."
    Attending Quotation: "Lung Bases: No focal consolidation. Scattered subcentimeter nodules likely incidental."
Explanation: While the attending quotation includes new information, it is not altering the confidence of the diagnosis, and should not be included.

Example Output 1:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
"Pancreas","There is a mass in the right lower quadrant, likely representing an appendiceal abscess.","There is a soft tissue density in the right lower quadrant, which may represent an appendiceal abscess; correlation with clinical findings is recommended.","7","The attending softened the diagnostic certainty and emphasized the need for clinical correlation."

Example Output 2:
"Section","Resident Report","Attending Report","Difference Type","Explanation"
'''

OLLAMA_MODEL = ["deepseek-r1:70b", "llama3.3:latest", "llama3.2-vision:90b", "gemma3:27b"]
OPENAI_MODEL = [ "gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"]

model_options = ["--Select--"] + OLLAMA_MODEL + OPENAI_MODEL
output_options = ["--Select--", "Paragraph Output", "Table Output"]
agent_format = ["--Select--", "Single Agent", "Multi-Agent"]


def strip_llm_wrappers(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Remove all <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove fenced code blocks like ```json\n...\n```
    text = re.sub(r"```(?:\w+)?\n(.*?)```", r"\1", text, flags=re.DOTALL)

    # Remove triple single or double quotes
    text = re.sub(r"'''(.*?)'''", r"\1", text, flags=re.DOTALL)
    text = re.sub(r'"""(.*?)"""', r"\1", text, flags=re.DOTALL)

    # Remove ~~~ fenced blocks
    text = re.sub(r"~~~(?:\w+)?\n(.*?)~~~", r"\1", text, flags=re.DOTALL)

    # Final cleanup
    return text.strip()

def string2df(response: str):
    reader = csv.DictReader(StringIO(response))
    data = list(reader)
    df = pd.DataFrame(data)

    return df

st.set_page_config(page_title="Report Comparison Tool", layout="wide")

st.title("Report Comparison Tool")

if "results" not in st.session_state:
    st.session_state["results"] = []

# --- Dropdown Menus ---
model = st.selectbox("Choose a Model:", model_options, index=0)
output_type = st.selectbox("Choose Output Format:", output_options, index=0)
output_agent_style = st.selectbox("Choose Agent Style:", agent_format, index=0)

output_setting = output_type == "Paragraph Output"
output_agent_setting = output_agent_style == "Single Agent"
# --- Prompt Editing Box ---
default_prompt = SYSTEM_PROMPT_PARAGRAPH if output_setting else SYSTEM_PROMPT_TABLE
multi_agent_prompts = [None, None, None, None, None, None, None]

if not output_type == "--Select--" and output_agent_setting:
    edited_prompt = st.text_area("Prompt Editing", value=default_prompt, height=300)
elif not output_type == "--Select--" and not output_agent_style == "--Select--":
    multi_agent_prompts[0] = st.text_area("Prompt Editing – Scale 1 Agent", value = AGENT_PROMPT_1, height = 200)
    multi_agent_prompts[1] = st.text_area("Prompt Editing – Scale 2 Agent", value = AGENT_PROMPT_2, height = 200)
    multi_agent_prompts[2] = st.text_area("Prompt Editing – Scale 3 Agent", value = AGENT_PROMPT_3, height = 200)
    multi_agent_prompts[3] = st.text_area("Prompt Editing – Scale 4 Agent", value = AGENT_PROMPT_4, height = 200)
    multi_agent_prompts[4] = st.text_area("Prompt Editing – Scale 5 Agent", value = AGENT_PROMPT_5, height = 200)
    multi_agent_prompts[5] = st.text_area("Prompt Editing – Scale 6 Agent", value = AGENT_PROMPT_6, height = 200)
    multi_agent_prompts[6] = st.text_area("Prompt Editing – Scale 7 Agent", value = AGENT_PROMPT_7, height = 200)

# --- Text Inputs Side-by-Side ---
col1, col2 = st.columns(2)

with col1:
    resident_text = st.text_area("Resident Report", height=300)

with col2:
    attending_text = st.text_area("Attending Revised Report", height=300)

# --- Button Enable Condition ---
button_disabled = (
    model == "--Select--" or
    output_type == "--Select--" or
    output_agent_style == "--Select--" or
    not resident_text.strip() or
    not attending_text.strip()
)

# --- Analyze Button ---
if st.button("Analyze", disabled=button_disabled):
    response = ""
    columns = ["Section", "Resident Report", "Attending Report", "Difference Type", "Explanation"]
    multi_df = pd.DataFrame(columns=columns)
    text = "Resident Report:\n" + resident_text + "\n\nAttending Report:\n" + attending_text

    if output_agent_setting:
        system_prompt = edited_prompt
        if model in OLLAMA_MODEL:
            response = extract_Ollama(system_prompt,text,model)
        else:
            response = asyncio.run(extract_OpenAI(system_prompt,text,model))
        
        result = strip_llm_wrappers(response)

        st.session_state["results"].append({
            "Model": model,
            "Resident Note": resident_text.strip(),
            "Attending Note": attending_text.strip(),
            "Output": result.strip(),
        })

        # Display the result
        if output_setting:
            st.write(result)
        else:
            # print(result)
            df = string2df(result)
            st.dataframe(df)
    else:
        model_type = model in OLLAMA_MODEL
        for agent_prompt in multi_agent_prompts:
            agent_response = None
            if model_type:
                agent_response = extract_Ollama(agent_prompt,text,model)
            else:
                agent_response = asyncio.run(extract_OpenAI(agent_prompt,text,model))
            clean = strip_llm_wrappers(agent_response)
            df = string2df(clean)

            multi_df = pd.concat([multi_df,df], ignore_index=True)
        
        if output_setting:
            first_part = ""
            second_part = ""
            if model_type:
                first_part = extract_Ollama(SYSTEM_PROMPT_PARAGRAPH_MULTI,text,model)
            else:
                first_part = asyncio.run(extract_OpenAI(SYSTEM_PROMPT_PARAGRAPH_MULTI,text,model))
            
            first_part = strip_llm_wrappers(first_part)

            for _, row in multi_df.iterrows():             
                second_part += f"\n\n{row["Difference Type"]}: \"{row["Resident Report"]}\" --> \"{row["Attending Report"]}\"\n\n{row["Explanation"]}"
            
            resp = first_part + second_part
            st.write(resp)

            st.session_state["results"].append({
                "Model": model,
                "Resident Note": resident_text.strip(),
                "Attending Note": attending_text.strip(),
                "Output": resp.strip(),
            })
        else:
            st.dataframe(multi_df)

            st.session_state["results"].append({
                "Model": model,
                "Resident Note": resident_text.strip(),
                "Attending Note": attending_text.strip(),
                "Output": multi_df.to_csv(index=False).strip(),
            })

# --- Generate download if there is anything to export ---
export_columns = ["Number", "Model", "Resident Note", "Attending Note", "Output"]

if "results" in st.session_state and st.session_state["results"]:
    df_to_save = pd.DataFrame(st.session_state["results"])
    df_to_save.insert(0, "Number", range(1, len(df_to_save) + 1))
else:
    df_to_save = pd.DataFrame(columns=export_columns)

csv_buffer = StringIO()
df_to_save.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

# --- Render download button immediately ---
st.download_button(
    label="📥 Download All Results as CSV",
    data=csv_data,
    file_name="comparison_outputs.csv",
    mime="text/csv"
)