from app.config.settings import Settings, get_settings


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.app_name == "agentic-true-vectorless-rag"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.retrieval_top_k == 10
    assert settings.retrieval_max_pages == 20
    assert settings.ocr_enabled is True


def test_cors_allowed_origins_list_default() -> None:
    settings = Settings()

    assert settings.cors_allowed_origins_list == [
        "http://localhost:3000"
    ]


def test_cors_allowed_origins_list_parses_multiple() -> None:
    settings = Settings(
        cors_allowed_origins=(
            "https://my-app.vercel.app, http://localhost:3000"
        )
    )

    assert settings.cors_allowed_origins_list == [
        "https://my-app.vercel.app",
        "http://localhost:3000",
    ]


def test_cors_allowed_origins_list_ignores_blank_entries() -> None:
    settings = Settings(
        cors_allowed_origins="https://my-app.vercel.app,,  ,"
    )

    assert settings.cors_allowed_origins_list == [
        "https://my-app.vercel.app"
    ]