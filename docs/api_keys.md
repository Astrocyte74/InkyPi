
# Storing API Keys

Certain plugins, like the AI Image plugin, require API credentials to function. These credentials must be stored in a .env file located at the root of the project. Once you have your API token, follow these steps:

1. SSH into your Raspberry Pi and navigate to the InkyPi directory:
    ```bash
    cd InkyPi
    ```
2. Create or edit the .env file using your preferred text editor (e.g., vi, nano):
    ```bash
    vi .env
    ```
3. Add your API keys following format, with one line per key:
    ```
    PLUGIN_KEY=your-key
    ```
4. Save the file and exit the editor

## OpenAI Key

Required for the AI Image and AI Text Plugins

- Login or create an account on the [OpenAI developer platform](https://platform.openai.com/docs/overview)
- Create a secret key from the API Keys tab in the Settings page
    - It is recommended to set up Auto recharge (found in the "Billing" tab)
    - Optionally set a Budget Limit in the Limits tab
- Store your key in the .env file with the key `OPEN_AI_SECRET`
    ```
    OPEN_AI_SECRET=your-key
    ```

## Open Weather Map Key

Required for the Weather Plugin

- Login or create an account on [OpenWeatherMap](https://home.openweathermap.org/users/sign_in)
    - Verify your email after signing up
- The weather plugin uses the [One Call API 3.0](https://openweathermap.org/price) which requires a subscription but is free for up to 1,000 requests per day.
    - Subscribe at [One Call API 3.0 Subscription](https://home.openweathermap.org/subscriptions/billing_info/onecall_30/base?key=base&service=onecall_30)
    - Follow the instructions to complete the subscription.
    - Navigate to [Your Subscriptions](https://home.openweathermap.org/subscriptions) and set "Calls per day (no more than)" to 1,000 to avoid exceeding the free limit
- Store your API key in the .env file with the key `OPEN_WEATHER_MAP_SECRET`
    ```
    OPEN_WEATHER_MAP_SECRET=your-key
    ```

## NASA Astronomy Picture Of the Day Key

Required for the APOD Plugin

- Request an API key on [NASA APIs](https://api.nasa.gov/)
   - Fill your First name, Last name, and e-mail address
- The APOD plugin uses the [NASA APIs](https://api.nasa.gov/)
   - Free for up to 1,000 requests per hour
- Store your API key in the .env file with the key `NASA_SECRET`
    ```
    NASA_SECRET=your-key
    ```

## Unsplash Key

Required for the Unsplash Plugin
 
- Register an account from https://unsplash.com/developers 
- Go to https://unsplash.com/oauth/applications 
- Create an app and open it
- Your KEY is listed as `Access Key`
- Store your API key in the .env file with the key `UNSPLASH_ACCESS_KEY`
    ```
    UNSPLASH_ACCESS_KEY=your-key
    ```

## GitHub Key

Required for the GitHub Plugin

- Login to your Github profile https://github.com/settings/profile
- Under Developer Settings, create a new Personal access token (classic)
- Assign the `read:user` scope and generate the token
- Store your API key in the .env file with the key `GITHUB_SECRET`
    ```
    GITHUB_SECRET=your-key
    ```

## Telegram Bot Token

Optional for remote updates via Telegram.

- Talk to [@BotFather](https://t.me/BotFather) to create a new bot and copy the token.
- Add the token to `/usr/local/inkypi/.env`:
    ```
    TELEGRAM_BOT_TOKEN=123456:ABC-your-token
    TELEGRAM_ALLOWED_IDS=123456789   # optional, comma-separated list of Telegram user IDs
    ```
- Restart the InkyPi service: `sudo systemctl restart inkypi.service`
- If `TELEGRAM_ALLOWED_IDS` is empty, the bot responds to any chat. Add your personal ID to restrict usage.
