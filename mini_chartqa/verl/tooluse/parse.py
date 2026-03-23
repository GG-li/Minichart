class Parser:
    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        oring_content = response.replace("\_", "_")
        content = oring_content.replace("\\", "")
        
        try:
            fence_starts = ["```python", "```py", "```python3"]
            start_pos = -1
            start_tag = None
            for tag in fence_starts:
                pos = content.find(tag)
                if pos != -1 and (start_pos == -1 or pos < start_pos):
                    start_pos = pos
                    start_tag = tag

            if start_pos == -1 or start_tag is None:
                extracted_lines = []
                for line in content.splitlines():
                    if "focus_on_" in line and "(" in line:
                        start = line.find("focus_on_")
                        extracted_lines.append(line[start:].strip())
                if extracted_lines:
                    prog = "\n".join(extracted_lines) + "\n"
                    try:
                        compile(prog, "prog.py", "exec")
                        return {'status': True, 'content': prog, 'message': 'Parsing succeeded.', 'error_code': ''}
                    except Exception as err:
                        return {'status': True, 'content': prog, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}
                return {'status': False, 'content': content, 'message': 'No tool call', 'error_code': 'NOTOOL'}

            content = content[start_pos + len(start_tag):]
            end_pos = content.find("```")
            if end_pos == -1:
                for marker in ["\nANSWER:", "\nFINAL ANSWER:", "\nTERMINATE", "\nOBSERVATION:", "\nACTION"]:
                    cut = content.find(marker)
                    if cut != -1:
                        content = content[:cut]
                        break
            else:
                content = content[:end_pos]

            if len(content.strip()) > 0:
                compile(content, "prog.py", "exec")
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'error_code': ''}
            return {'status': False, 'content': content, 'message': "The content is empty, or it failed to parse the content correctly.", 'error_code': 'unknown'}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}
        
    def trim_to_action_end(self, text):
        last_code_block_start = text.rfind("```")
        if last_code_block_start == -1:
            return text  # no code block found
        # Now find the preceding ``` that starts the block
        preceding_code_block_start = text.rfind("```", 0, last_code_block_start)
        if preceding_code_block_start == -1:
            return text  # malformed block
        return text[:last_code_block_start + 3]  # include closing ```
    
def main():
    parser = Parser()
    
    # testing 1
    program = """Thought: I thought a lot and here is what I am thinking.\nAction:```python\n"""
    program += """def solve():\n"""
    program += """    output0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    program += """HELLO WORLD"""
    #print(program)
    results = parser.parse(program)
    #print(results)

    print(parser.trim_to_action_end(program))
    
    print("\n\n-----------------------------------\n\n")
    
    # testing 2
    program = """Thought: I thought a lot and here is what I am thinking.\nAction:```python\n"""
    program += """def solve():\n"""
    program += """    aha\noutput0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    print("\n\n-----------------------------------\n\n")
    
    # testing 3
    program = """Thought: I thought a lot and here is what I am thinking.\nAction: No need"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    
if __name__ == '__main__':
    main()
