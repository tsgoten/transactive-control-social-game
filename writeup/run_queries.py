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
                    },
            "norl_simple": {"$or": [
                        {"config.tag": "simple_norl10"},]
                    },
            "norl_medium": {"$or": [
                        {"config.tag": "medium_norl10"},]
                    },
            "norl_complex": {"$or": [
                        {"config.tag": "complex_norl10"},]
                    },
}