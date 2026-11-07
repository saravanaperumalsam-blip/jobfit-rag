# Retrieval evaluation

Run on 2026-11-06 with 34 questions (24 keyword, 10 paraphrase).

| embeddings | chunk_size | overlap | chunks | hit@1 | hit@3 | mrr | hit@1 keyword | hit@1 paraphrase |
|---|---|---|---|---|---|---|---|---|
| minilm | 40 | 10 | 94 | 0.79 | 0.94 | 0.86 | 0.83 | 0.70 |
| minilm | 120 | 20 | 33 | 0.82 | 0.94 | 0.88 | 0.88 | 0.70 |
| minilm | 250 | 40 | 27 | 0.79 | 0.91 | 0.86 | 0.83 | 0.70 |
| tfidf | 40 | 10 | 94 | 0.88 | 0.94 | 0.91 | 0.96 | 0.70 |
| tfidf | 120 | 20 | 33 | 0.88 | 0.94 | 0.91 | 0.96 | 0.70 |
| tfidf | 250 | 40 | 27 | 0.85 | 0.94 | 0.90 | 0.96 | 0.60 |
| hybrid | 40 | 10 | 94 | 0.85 | 0.94 | 0.90 | 0.88 | 0.80 |
| hybrid | 120 | 20 | 33 | 0.85 | 0.94 | 0.90 | 0.88 | 0.80 |
| hybrid | 250 | 40 | 27 | 0.82 | 0.91 | 0.89 | 0.88 | 0.70 |

## Questions not ranked first

- [minilm, 40] Which role needs Excel with Power Query for a business placement in Slough? -> expected `akzonobel_placement_analyst`, got `cummins_business_analyst_placement`
- [minilm, 40] Which graduate scheme wants A-level Mathematics at grade A and insurance pricing work? -> expected `wtw_data_scientist_graduate`, got `aviva_ai_ml_graduate`
- [minilm, 40] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [minilm, 40] Which role does not accept Graduate or post-study work visas? -> expected `sparta_junior_data_consultant`, got `aviva_ai_ml_graduate`
- [minilm, 40] Where could I build software for precision measurement products in a rural part of England? -> expected `renishaw_software_placement`, got `atkinsrealis_data_intelligence_placement`
- [minilm, 40] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `aviva_data_engineering_graduate`
- [minilm, 40] Which job improves travel booking apps on phones? -> expected `expedia_mobile_intern`, got `expedia_applied_science_intern`
- [minilm, 120] Which programme needs a completed PhD or Master's and Bayesian inference? -> expected `faculty_data_science_fellowship`, got `arm_data_analytics_intern`
- [minilm, 120] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [minilm, 120] Which role does not accept Graduate or post-study work visas? -> expected `sparta_junior_data_consultant`, got `cgi_data_analysis_graduate`
- [minilm, 120] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [minilm, 120] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `cgi_applied_ai_graduate`
- [minilm, 120] Where would I help a chip design company check the quality of its products? -> expected `arm_data_analytics_intern`, got `akzonobel_data_science_placement`
- [minilm, 250] Which role needs Excel with Power Query for a business placement in Slough? -> expected `akzonobel_placement_analyst`, got `cummins_business_analyst_placement`
- [minilm, 250] Which internship builds AI agents with retrieval and tool calling? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [minilm, 250] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [minilm, 250] Which role does not accept Graduate or post-study work visas? -> expected `sparta_junior_data_consultant`, got `cgi_data_analysis_graduate`
- [minilm, 250] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [minilm, 250] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `cgi_applied_ai_graduate`
- [minilm, 250] Where would I help a chip design company check the quality of its products? -> expected `arm_data_analytics_intern`, got `atkinsrealis_data_intelligence_placement`
- [tfidf, 40] Which 10-week engineering internship is open to any STEM student? -> expected `rollsroyce_engineering_internship`, got `bloomberg_analytics_sales_internship`
- [tfidf, 40] Which opportunity is about smarter roads and railways? -> expected `wsp_intelligent_infrastructure_graduate`, got `(no match)`
- [tfidf, 40] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `atkinsrealis_data_intelligence_placement`
- [tfidf, 40] Where would I help a chip design company check the quality of its products? -> expected `arm_data_analytics_intern`, got `cgi_data_analysis_graduate`
- [tfidf, 120] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [tfidf, 120] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [tfidf, 120] Which opportunity is about smarter roads and railways? -> expected `wsp_intelligent_infrastructure_graduate`, got `(no match)`
- [tfidf, 120] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `atkinsrealis_data_intelligence_placement`
- [tfidf, 250] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [tfidf, 250] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [tfidf, 250] Which opportunity is about smarter roads and railways? -> expected `wsp_intelligent_infrastructure_graduate`, got `(no match)`
- [tfidf, 250] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `atkinsrealis_data_intelligence_placement`
- [tfidf, 250] Where would I help a chip design company check the quality of its products? -> expected `arm_data_analytics_intern`, got `cgi_data_analysis_graduate`
- [hybrid, 40] Which role needs Excel with Power Query for a business placement in Slough? -> expected `akzonobel_placement_analyst`, got `cummins_business_analyst_placement`
- [hybrid, 40] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [hybrid, 40] Which role does not accept Graduate or post-study work visas? -> expected `sparta_junior_data_consultant`, got `aviva_ai_ml_graduate`
- [hybrid, 40] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [hybrid, 40] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `sparta_junior_data_consultant`
- [hybrid, 120] Which programme needs a completed PhD or Master's and Bayesian inference? -> expected `faculty_data_science_fellowship`, got `arm_data_analytics_intern`
- [hybrid, 120] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [hybrid, 120] Which role does not accept Graduate or post-study work visas? -> expected `sparta_junior_data_consultant`, got `aviva_data_engineering_graduate`
- [hybrid, 120] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [hybrid, 120] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `sparta_junior_data_consultant`
- [hybrid, 250] Which role needs Excel with Power Query for a business placement in Slough? -> expected `akzonobel_placement_analyst`, got `cummins_business_analyst_placement`
- [hybrid, 250] Which internship is only for penultimate-year undergraduates at a TV company? -> expected `sky_data_science_internship`, got `roku_ai_engineering_internship`
- [hybrid, 250] Which role does not accept Graduate or post-study work visas? -> expected `sparta_junior_data_consultant`, got `aviva_data_engineering_graduate`
- [hybrid, 250] Which internship lets me build assistants that investigate failures and automate engineering workflows? -> expected `roku_ai_engineering_internship`, got `arm_data_analytics_intern`
- [hybrid, 250] Which role trains people with no degree requirement before sending them to client projects? -> expected `sparta_data_consultant`, got `cgi_applied_ai_graduate`
- [hybrid, 250] Where would I help a chip design company check the quality of its products? -> expected `arm_data_analytics_intern`, got `cgi_data_analysis_graduate`
