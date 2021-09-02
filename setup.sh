mkdir -p ~/.streamlit

echo "[server]
headless = true
port = $PORT
enableCORS = false

[theme]
primaryColor='#e27936'
backgroundColor='#828282'
secondaryBackgroundColor='#6d6d6d'
textColor='#ffffff'
font='sans serif'

" > ~/.streamlit/config.toml
