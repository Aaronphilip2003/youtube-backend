from django.shortcuts import render
from django.http import HttpResponse
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from django.http import JsonResponse
from langchain.embeddings import GooglePalmEmbeddings
from transformers import pipeline
from django.http import JsonResponse
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse as urlparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
import json
from langchain.llms.huggingface_hub import HuggingFaceHub
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
import re
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import UnstructuredPowerPointLoader


@csrf_exempt
def fileproc(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES['document']
            save_path = './query/uploaded_documents/'  # Your desired folder path
            file_name = uploaded_file.name
            file_path = os.path.join(save_path, file_name)

            # Save the file to the specified path
            with open(file_path, 'wb') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            print(file_name, "saved in the uploaded_documents directory")

            # Your additional processing logic goes here
            base_name, extension = os.path.splitext(file_name)

            if (extension == ".txt"):
                print("This is txt")
                with open(f'./query/uploaded_documents/{file_name}', 'r') as file:
                    text = file.read()
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700, chunk_overlap=0, length_function=len)
                chunks = text_splitter.split_text(text)
                docs = text_splitter.create_documents(chunks)

                # Convert chunks to embeddings and save as FAISS file
                embedding = GooglePalmEmbeddings(
                    google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
                vdb_chunks_HF = FAISS.from_documents(docs, embedding=embedding)
                vdb_chunks_HF.save_local(
                    f'./query/uploaded_documents/', index_name=f"{base_name}")

            elif (extension == ".pdf"):
                print("This is pdf")
                print(file_name)
                docs = PDFMinerLoader(
                    f"./query/uploaded_documents/{file_name}").load()
                print(docs)
                # docs = text_splitter.create_documents(chunks)
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700, chunk_overlap=0, length_function=len)
                chunks = text_splitter.split_documents(docs)
                # Convert chunks to embeddings and save as FAISS file
                # embedding = GooglePalmEmbeddings(google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
                embedding2 = HuggingFaceEmbeddings()
                vdb_chunks_HF = FAISS.from_documents(
                    docs, embedding=embedding2)
                vdb_chunks_HF.save_local(
                    f'./query/uploaded_documents/', index_name=f"{base_name}")

            elif (extension == ".pptx" or "ppt"):
                print("This is ppt")
                loader = UnstructuredPowerPointLoader(
                    f"./query/uploaded_documents/{file_name}", mode="slides")
                print(f"/query/uploaded_documents/{file_name}")
                docs = loader.load()
                print(docs[0])
                # text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0, length_function=len)
                # chunks = text_splitter.split_documents(docs)
                # # Convert chunks to embeddings and save as FAISS file
                # embedding = GooglePalmEmbeddings(google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
                # vdb_chunks_HF = FAISS.from_documents(docs, embedding=embedding)
                # vdb_chunks_HF.save_local(f'./query/uploaded_documents/', index_name=f"{base_name}")

            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'error': str(e)})
    else:
        return JsonResponse({'status': 'error', 'error': 'Invalid request method'})


def answerfile(request):
    user_message = request.GET.get('query', '')
    file_name = request.GET.get('name', '')
    base_name, extension = os.path.splitext(file_name)
    embedding2 = GooglePalmEmbeddings(
        google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
    # embedding2 = HuggingFaceEmbeddings()
    # Adjust the path to your FAISS index directory
    db = FAISS.load_local("./query/uploaded_documents/",
                          embedding2, index_name=f"{base_name}")
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={
                         "temperature": 0.1, "max_length": 65536, "min_length": 512}, huggingfacehub_api_token="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")
    # google/flan-t5-xxl
    # tiiuae/falcon-7b-instruct
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(user_message)
    response = chain.run(input_documents=docs, question=user_message)

    # Add CORS headers to the response
    response = JsonResponse({'response': response})
    response['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    return response


def extract_video_id(url):
    print("This is the url", url)
    # Parse URL
    url_data = urlparse.urlparse(url)
    # Extract video id
    video_id = urlparse.parse_qs(url_data.query)['v'][0]
    print(f"this is the video id {video_id}")
    return video_id


def say_hello(request):
    return HttpResponse("Hello")


def yt(request):
    # Get YouTube URL from request parameters
    url = request.GET.get('url', '')
    # Extract video id
    video_id = extract_video_id(url)

    embedding2 = GooglePalmEmbeddings(
        google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
    vdb_chunks_HF = FAISS.load_local(
        f"./query/vdb_chunks_HF/", embedding2, index_name=f"index{video_id}")
    query = request.GET.get('query', '')
    ans = vdb_chunks_HF.as_retriever().get_relevant_documents(query)
    answers = [doc.page_content for doc in ans]

    # Add CORS headers to the response
    response = JsonResponse({'answers': answers})
    response['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    return response


def llm_answering(request):
    # Assuming you're using Django for web development
    url = request.GET.get('url', '')
    query = request.GET.get('query', '')
    print(query)
    # Validate if both 'url' and 'query' parameters are present
    if not url or not query:
        return HttpResponse("Both 'url' and 'query' parameters are required.")
    video_id = extract_video_id(url)
    embedding2 = GooglePalmEmbeddings(
        google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
    # Adjust the path to your FAISS index directory
    db = FAISS.load_local("./query/vdb_chunks_HF/",
                          embedding2, index_name=f"index{video_id}")
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={
                         "temperature": 0.1, "max_length": 65536, "min_length": 32768}, huggingfacehub_api_token="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")
    # google/flan-t5-xxl
    # tiiuae/falcon-7b-instruct
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    # Add CORS headers to the response
    response = JsonResponse({'response': response})
    response['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    return response


def process_youtube_video(request):
    print(request)
    # Get YouTube URL from request parameters
    url = request.GET.get('query', '')
    print("This is the url", url)
    # Extract video id
    video_id = extract_video_id(url)

    try:
        # Fetch the captions
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en'])
        captions = transcript.fetch()
        captions_text = [caption['text'] for caption in captions]

        # Open a text file in write mode
        with open(f'./query/vdb_chunks_HF/{video_id}_captions.txt', 'w') as f:
            for caption in captions:
                # Write the caption to the text file
                f.write(caption['text'] + '\n')
        print("Captions saved successfully")

        # Load text file
        with open(f'./query/vdb_chunks_HF/{video_id}_captions.txt', 'r') as file:
            text = file.read()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=0, length_function=len)
        chunks = text_splitter.split_text(text)

        # Create documents from chunks
        docs = text_splitter.create_documents(chunks)

        # Convert chunks to embeddings and save as FAISS file
        embedding = GooglePalmEmbeddings(
            google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
        vdb_chunks_HF = FAISS.from_documents(docs, embedding=embedding)
        vdb_chunks_HF.save_local(f'./query/vdb_chunks_HF/',
                                 index_name=f"index{video_id}")

        return JsonResponse({'transcript': captions_text, 'status': 'success'})

    except Exception as e:
        print("An error occurred:", e)
        return JsonResponse({'error': "An error occurred while fetching the captions."})


def summarize_video(request):
    # Get YouTube URL from request parameters
    url = request.GET.get('url', '')
    # Extract video id
    video_id = extract_video_id(url)
    # Set the question to "summarize the video"
    query = "summarize the video"

    embedding2 = GooglePalmEmbeddings(
        google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
    # Adjust the path to your FAISS index directory
    db = FAISS.load_local("./query/vdb_chunks_HF/",
                          embedding2, index_name=f"index{video_id}")
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={
                         "temperature": 0.1, "max_length": 65536, "min_length": 32768}, huggingfacehub_api_token="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")
    # google/flan-t5-xxl
    # tiiuae/falcon-7b-instruct
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    # Add CORS headers to the response
    response = JsonResponse({'response': response})
    response['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    return response


def question_generation(request):
    # Get YouTube URL from request parameters
    url = request.GET.get('url', '')
    # Extract video id
    video_id = extract_video_id(url)
    # Set the question to "summarize the video"
    query = "Generate 5 questions from the video and generate it based on the context"

    embedding2 = GooglePalmEmbeddings(
        google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
    # Adjust the path to your FAISS index directory
    db = FAISS.load_local("./query/vdb_chunks_HF/",
                          embedding2, index_name=f"index{video_id}")
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={
                         "temperature": 0.1, "max_length": 65536, "min_length": 32768}, huggingfacehub_api_token="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")
    # google/flan-t5-xxl
    # tiiuae/falcon-7b-instruct
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    # Add CORS headers to the response
    response = JsonResponse({'response': response})
    response['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    return response


# http://127.0.0.1:8000/query/yt/?query=omega&url=https://www.youtube.com/watch?v=7kcWV6zlcRU&list=PLUl4u3cNGP62esZEwffjMAsEMW_YArxYC&index=5&ab_channel=MITOpenCourseWare
# http://127.0.0.1:8000/query/ytvid/?url=https://www.youtube.com/watch?v=IcmzF1GT1Qw
# http://127.0.0.1:8000/query/llm/?query=what+is+omega&url=https://www.youtube.com/watch?v=7kcWV6zlcRU&list=PLUl4u3cNGP62esZEwffjMAsEMW_YArxYC&index=5&ab_channel=MITOpenCourseWare
# http://127.0.0.1:8000/query/llm/?query=what+does+he+say+about+beethoven+?&url=https://www.youtube.com/watch?v=IcmzF1GT1Qw&ab_channel=Vienna
# https://www.youtube.com/watch?v=Tuw8hxrFBH8
# http://127.0.0.1:8000/query/llm/summarize_video/?url=${trainingUrl}
