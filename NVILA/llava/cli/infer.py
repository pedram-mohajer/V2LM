# import argparse
# import importlib.util
# import json
# import os

# from pydantic import BaseModel
# from termcolor import colored

# import llava
# from llava import conversation as clib
# from llava.media import Image, Video
# from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat


# def get_schema_from_python_path(path: str) -> str:
#     schema_path = os.path.abspath(path)
#     spec = importlib.util.spec_from_file_location("schema_module", schema_path)
#     schema_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(schema_module)

#     # Get the Main class from the loaded module
#     Main = schema_module.Main
#     assert issubclass(
#         Main, BaseModel
#     ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
#     return Main.schema_json()


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", "-m", type=str, required=True)
#     parser.add_argument("--conv-mode", "-c", type=str, default="auto")
#     parser.add_argument("--text", type=str)
#     parser.add_argument("--media", type=str, nargs="+")
#     parser.add_argument("--json-mode", action="store_true")
#     parser.add_argument("--json-schema", type=str, default=None)
#     args = parser.parse_args()

#     # Convert json mode to response format
#     if not args.json_mode:
#         response_format = None
#     elif args.json_schema is None:
#         response_format = ResponseFormat(type="json_object")
#     else:
#         schema_str = get_schema_from_python_path(args.json_schema)
#         print(schema_str)
#         response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

#     # Load model
#     model = llava.load(args.model_path)

#     # Set conversation mode
#     clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

#     # Prepare multi-modal prompt
#     prompt = []
#     if args.media is not None:
#         for media in args.media or []:
#             if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
#                 media = Image(media)
#             elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
#                 media = Video(media)
#             else:
#                 raise ValueError(f"Unsupported media type: {media}")
#             prompt.append(media)
#     if args.text is not None:
#         prompt.append(args.text)

#     # Generate response
#     response = model.generate_content(prompt, response_format=response_format)
#     print(colored(response, "cyan", attrs=["bold"]))


# if __name__ == "__main__":
#     main()

##########################################################################################

# import argparse
# import importlib.util
# import os
# import csv
# from pydantic import BaseModel
# from termcolor import colored
# import llava
# from llava import conversation as clib
# from llava.media import Video
# from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat


# def get_schema_from_python_path(path: str) -> str:
#     """Load and return JSON schema from a Python file."""
#     schema_path = os.path.abspath(path)
#     spec = importlib.util.spec_from_file_location("schema_module", schema_path)
#     schema_module = importlib.util.module_from_spec
#     spec.loader.exec_module(schema_module)

#     Main = schema_module.Main
#     assert issubclass(
#         Main, BaseModel
#     ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
#     return Main.schema_json()


# def process_videos(csv_file, model, response_format, output_csv, text_prompt):
#     """Reads video paths from a CSV file, processes them with the model, and saves results to another CSV."""
#     results = []

#     with open(csv_file, mode="r", encoding="utf-8") as file:
#         reader = csv.reader(file)
#         header = next(reader)
#         if "Filename" not in header:
#             raise ValueError("CSV file must have a 'Filename' column.")

#         for row in reader:
#             video_path = row[0].strip()

#             if not os.path.exists(video_path):
#                 print(colored(f"Skipping missing file: {video_path}", "red"))
#                 continue

#             prompt = [Video(video_path)]
#             if text_prompt:
#                 prompt.append(text_prompt)

#             response = model.generate_content(prompt, response_format=response_format)
#             #print(video_path,"----->", response)
#             results.append([video_path, response])

#             print(colored(f"Processed: {video_path}", "green"))

#     with open(output_csv, mode="w", encoding="utf-8", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Filename", "output"])
#         writer.writerows(results)

#     print(colored(f"Results saved to {output_csv}", "cyan", attrs=["bold"]))


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to the LLaVA model.")
#     parser.add_argument("--csv-file", type=str, required=True, help="CSV file containing video paths.")
#     parser.add_argument("--output-csv", type=str, default="output.csv", help="Output CSV file.")
#     parser.add_argument("--conv-mode", "-c", type=str, default="auto", help="Conversation mode.")
#     parser.add_argument("--json-mode", action="store_true", help="Enable JSON mode.")
#     parser.add_argument("--json-schema", type=str, default=None, help="Path to JSON schema file.")
#     parser.add_argument("--text", type=str, default=None, help="Optional text prompt for all videos.")

#     args = parser.parse_args()

#     if not args.json_mode:
#         response_format = None
#     elif args.json_schema is None:
#         response_format = ResponseFormat(type="json_object")
#     else:
#         schema_str = get_schema_from_python_path(args.json_schema)
#         response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

#     print(colored("Loading model...", "yellow"))
#     model = llava.load(args.model_path)
#     clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

#     process_videos(args.csv_file, model, response_format, args.output_csv, args.text)


# if __name__ == "__main__":
#     main()

##########################################################################################


import argparse
import importlib.util
import os
import csv
from pydantic import BaseModel
from termcolor import colored
import llava
from llava import conversation as clib
from llava.media import Video, Image  # Import Image processing class
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
import time
# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".ppm"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}



def get_schema_from_python_path(path: str) -> str:
    """Load and return JSON schema from a Python file."""
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


def process_files(csv_file, model, response_format, output_csv, text_prompt):
    """Reads file paths (images/videos) from a CSV, processes them, and saves results to another CSV."""
    results = []
    start_time = time.time()
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        if "Filename" not in header:
            raise ValueError("CSV file must have a 'Filename' column.")

        for row in reader:
            file_path = row[0].strip()

            if not os.path.exists(file_path):
                print(colored(f"Skipping missing file: {file_path}", "red"))
                continue

            file_ext = os.path.splitext(file_path)[1].lower()

            # Determine if it's an image or a video
            if file_ext in VIDEO_EXTENSIONS:
                prompt = [Video(file_path)]
                media_type = "Video"
            elif file_ext in IMAGE_EXTENSIONS:
                prompt = [Image(file_path)]
                media_type = "Image"
            else:
                print(colored(f"Unsupported file type: {file_path}", "yellow"))
                continue

            if text_prompt:
                prompt.append(text_prompt)

            response = model.generate_content(prompt, response_format=response_format)
            results.append([file_path, response])

            print(colored(f"Processed {media_type}: {file_path}", "green"))
    end_time = time.time()
    total_time = end_time - start_time
    num_files = len(results)
    average_time = total_time / num_files if num_files > 0 else 0
    print(f"Average processing time per file: {average_time:.2f} seconds")



    # Save results to output CSV
    with open(output_csv, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Output"])
        writer.writerows(results)

    print(colored(f"Results saved to {output_csv}", "cyan", attrs=["bold"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to the LLaVA model.")
    parser.add_argument("--csv-file", type=str, required=True, help="CSV file containing file paths (images/videos).")
    parser.add_argument("--output-csv", type=str, default="output.csv", help="Output CSV file.")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto", help="Conversation mode.")
    parser.add_argument("--json-mode", action="store_true", help="Enable JSON mode.")
    parser.add_argument("--json-schema", type=str, default=None, help="Path to JSON schema file.")
    parser.add_argument("--text", type=str, default=None, help="Optional text prompt for all media files.")

    args = parser.parse_args()

    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    print(colored("Loading model...", "yellow"))
    model = llava.load(args.model_path)
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    process_files(args.csv_file, model, response_format, args.output_csv, args.text)


if __name__ == "__main__":
    main()
