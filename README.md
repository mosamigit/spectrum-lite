Steps to run the model

- Install dependencies 
  - Make sure Python 3.9 version has installed on system.
 
    ```
    python -m pip install -r requirements.txt
    ```

- Set environment variables 
    - Copy the `env.template` file to `.env` file
    - Set values for environment variables

- Run as a standalone script

    ```
    python spectrum_lite.py 
    ```

- Run as a cron job

    ```
    python spectrum_lite_cron.py 
    ```


