# Installation


1. Install **ffmpeg** on your system. Some packages rely on it and we generally recommend ffmpeg because it is very useful for dealing with video data.
    - Open a terminal window/command prompt

        ??? note "Opening a terminal window/command prompt"
            === "Windows"
                1. Click the Windows key + R
                2. Type `cmd` and click *OK* to open a terminal window.
            === "MacOS"
                1. Click the Launchpad icon.
                2. Type `Terminal` in the search field and click on *Terminal* to open it.
            === "Linux"
                You know what to do :)

    - Type `ffmpeg -version` and click *Enter* <br>
        *If this command fails for some reason, make sure you install ffmpeg first*

        ??? note "Installing ffmpeg"
            === "Windows"
                1. Open the [ffmpeg download page](https://ffmpeg.org/download.html).
                2. Under the "Get packages & executable files" section, click on the Windows logo.
                3. You will be redirected to a page with various builds. Click on the link for the "Windows builds from gyan.dev".
                4. Scroll down to the "Release builds" section and download the `ffmpeg-release-essentials.zip` file.
                5. Extract the downloaded zip file and copy the `bin` subfolder to, for example, `C:\ffmpeg\bin`.
                6. Open the Start menu in Windows, search for "Environment Variables", and select "Edit the system environment variables".
                7. In the System Properties window, click on the "Environment Variables..." button.
                8. In the Environment Variables window, find the "Path" variable under the "System variables" section and select it. Click "Edit...".
                9. In the Edit Environment Variable window, click "New" and paste the path to the `bin` directory (e.g., `C:\ffmpeg\bin`). Click "OK" to close all windows.
                10. Verify that the installation is complete: Open a new (!) command prompt (cmd) and type `ffmpeg -version` and press `Enter`.
                11. If ffmpeg is installed correctly, you should see the version information for ffmpeg.
            === "MacOS"
                On MacOS you can use [homebrew](https://formulae.brew.sh/formula/ffmpeg) and type `brew install ffmpeg` into your terminal
            === "Linux"
                Instructions depend on your system, but please do not hesitate to reach out if you run into issues. 

2. Install Anaconda. <br>
     - Open your web browser and go to the official [Anaconda download page](https://www.anaconda.com/download).
     - Download and execute the Anaconda Installer for your operating system (Windows, macOS, or Linux). During installation, make sure you check the box **"Add Anaconda to PATH"**. This enables you to run `conda` commands in your terminal.
     - Restart your terminal.

3. Clone the ethograph repository. <br>
    Navigate to an easily accessible folder (e.g., `Documents`) and clone the repository.

    ??? note "Cloning the repository"
        === "Windows"
            1. Open File Explorer and navigate to a folder where you want to store ethograph (e.g., `Documents`).
            2. Click on the address bar, type `cmd`, and press *Enter* to open a terminal in that folder.
            3. Run the following command:
                ```bash
                git clone https://github.com/Akseli-Ilmanen/ethograph
                ```
            4. Wait for the download to complete. You will now have an `ethograph` folder in your chosen location.

            !!! tipp "Don't have Git installed?"
                If the `git` command is not recognized, you need to install Git first:

                1. Download Git from [git-scm.com](https://git-scm.com/download/win).
                2. Run the installer and follow the prompts (default options are fine).
                3. Restart your terminal and try the clone command again.

        === "MacOS / Linux"
            1. Open a terminal and navigate to a folder where you want to store ethograph:
                ```bash
                cd ~/Documents
                ```
            2. Clone the repository:
                ```bash
                git clone https://github.com/Akseli-Ilmanen/ethograph
                ```
            3. Navigate into the cloned folder:
                ```bash
                cd ethograph
                ```

            !!! tipp "Don't have Git installed?"
                === "MacOS"
                    Install Git using Homebrew:
                    ```bash
                    brew install git
                    ```
                === "Linux"
                    Install Git using your package manager:
                    ```bash
                    # Ubuntu/Debian
                    sudo apt install git

                    # Fedora
                    sudo dnf install git
                    ```

4. Install the ethograph conda environment. <br>

    Open a terminal and navigate to the ethograph folder:
    ```bash
        cd /path/to/ethograph
    ```
    Create the conda environment:
    ```bash
        conda env create -f environment.yml && conda activate ethograph && pip install -r requirements.txt
    ```
    Optionally, create a desktop shortcut:
    ```bash
        ethograph shortcut
    ```
    
 

5. Test it works. <br>

    ??? question "How to launch the GUI?"
        === "Windows"
            Double click on the new desktop shortcut.
            
            Or manually activate the conda environment and launch the GUI.
            ```bash
            conda activate ethograph
            ethograph launch
            ```      

        === "MacOS / Linux"

            Activate the conda environment and launch the GUI.
            ```bash
            conda activate ethograph
            ethograph launch
            ```      
