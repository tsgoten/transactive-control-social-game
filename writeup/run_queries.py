QUERIES = {
            "zeroshot_20" :   {"$or": [
                        {"config.tag": "medium_zeroshot_baseline_custom20_alt"},
                        {"config.tag": "medium_zeroshot_hnetagg_custom20_altembed"},
                        {"config.tag": "medium_hnetagg_custom20_alt"},]
                    },
            "zeroshot_all": {"$or": [
                        {"config.tag": "medium_zeroshot_baseline_custom20_alt"},
                        {"config.tag": "medium_zeroshot_hnetagg_custom20_altembed"},
                        {"config.tag": "medium_hnetagg_custom20_alt"},
                        {"config.tag": "medium_zeroshot_hnetagg_custom5_alt"},
                        {"config.tag": "medium_zeroshot_hnetagg_custom10_noembed"}]
                    }
}