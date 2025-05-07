🛠️ Project Setup Instructions

Follow these steps to set up the project on your local machine.

1. 📥 Clone the Repository

Open a terminal and navigate to the folder where you want to clone the repository:

` git clone https://github.com/your-username/data-analysis.git `

2. 🐍 Create a Virtual Environment
In the terminal (either the same terminal as before or VS Code), run:

` python3 -m venv venv `

3. ▶️ Activate the Virtual Environment

On macOS/Linux:
`source venv/bin/activate`

On Windows:
`venv\Scripts\activate`

4. 📦 Install Project Dependencies

With the virtual environment activated, install all required packages:

`pip install -r requirements.txt`

5. 🧠 Configure Python Interpreter in VS Code

Open the project in VS Code (code . if you’re in the terminal).

Open the Command Palette:

Cmd+Shift+P on macOS

Ctrl+Shift+P on Windows

Search for and select:

Python: Select Interpreter

Choose the interpreter from your virtual environment:

./venv/bin/python (macOS/Linux)

.\venv\Scripts\python.exe (Windows)
