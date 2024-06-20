# CustomerCareBot

CustomerCareBot is a Streamlit web application designed to automate customer service responses using AI-powered language models. It uses your knowledge base i.e (a companies data) to generate a response. In this implementation it uses past data between customers and customer care agents. It provides a convenient interface for generating automated responses based on customer inquiries or messages.

The data was extracted from Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual Assistants

## Features

- **Automated Response Generation:** Utilizes OpenAI's GPT-3.5-turbo model to generate context-aware responses to customer messages.
  
- **Conversation History:** Maintains a history of user interactions and bot responses for reference and continuity.
  
- **Easy-to-Use Interface:** Simple text area for entering customer messages and a button to generate responses quickly.
  
- **Customizable Settings:** Options to adjust response generation parameters like temperature and model selection.

## Installation

To run CustomerCareBot locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/Tobsky/CustomerCare_Bot.git
   cd your-repository

2. Install dependencies:

    ```bash
    pip install -r requirements.txt

3. Set up environment variables:

    Ensure you have set the necessary environment variables, such as OPENAI_API_KEY for OpenAI authentication.

4. Run the Streamlit app:

    ```bash
    streamlit run app.py

5. Open your web browser and go to http://localhost:8501 to view the app.

## Usage

1. Enter a customer message in the text area labeled "Customer message".

2. Press Ctrl+Enter button on your keyboard to trigger the AI model for generating a response.

3. The generated response will be displayed in the interface, along with the previous conversation history.

4. Use the conversation history to maintain context and continuity in customer interactions.

## Example

Here's how CustomerCareBot can be used:

1.  Customer Input:

    ```css
    Hello, I need help canceling an order I made.

2.  Generated Response:

    ```vbnet
    Assistant: I understand your concern. To cancel your order, please provide your order number and contact information so we can assist you further.

## Contributions

Contributions are welcome! If you have suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.