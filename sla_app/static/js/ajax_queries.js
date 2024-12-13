$(document).ready(function() {
    

    $('#rag-form button[type="submit"]').on('click', function(e){
        
        // Show the loader
        $('#loader').show();

        // Initialize result
        $('.result span').html("In waiting for a response ...")

        data_obj = {
            query: $('form textarea[name="query"]').val(),
            temperature: $('form input[name="temperature"]').val(),
            base_weight: $('form input[name="base_weight"]').val(),
            base_n: $('form input[name="base_n"]').val(),
            bm25_n: $('form input[name="bm25_n"]').val(),
            max_retries: $('form input[name="max_retries"]').val(),
            chat_model: $('form select[name="chat_model"]').val(),
            target: $('form select[name="target"]').val(),
            embedding_id: $('form select[name="embedding_id"]').val(),
            metric: $('form select[name="metric"]').val(),
            reranker: $('form select[name="reranker"]').val(),
            c_prompt: $('form textarea[name="c_prompt"]').val(),
            e_prompt: $('form textarea[name="e_prompt"]').val(),
            q_prompt: $('form textarea[name="q_prompt"]').val(),
        }
        
        console.log(data_obj)

        $.ajax({
            data: data_obj,
            type: 'POST',
            url: '/rag_system/',
            complete: function() {
                // Hide the loader
                $('#loader').hide();
            },
            success: function(data) {
                console.log(data)
                
                $('title').text(data.title);

                $('.response').html(data.response);

                $('form textarea[name="query"]').val(data.query);
                
                $('form input[name="temperature"]').val(data.temperature);

                $('form input[name="base_weight"]').val(data.base_weight);
                
                $('form input[name="base_n"]').val(data.base_n);

                $('form input[name="bm25_n"]').val(data.bm25_n);

                $('form input[name="max_retries"]').val(data.max_retries);

                $('form select[name="target"]').val(data.target);

                $('form select[name="chat_model"]').val(data.chat_model);

                $('form select[name="embedding_id"]').val(data.embedding_id);
                    
                $('form select[name="metric"]').val(data.metric);
                
                $('form select[name="reranker"]').val(data.reranker);

                $('form textarea[name="c_prompt"]').val(data.c_prompt);

                $('form textarea[name="e_prompt"]').val(data.e_prompt);

                $('form textarea[name="q_prompt"]').val(data.q_prompt);
                
                if(data.correct){

                    $('.result span').html("")
                    
                    $('.result span').html(`New Query: <p class="text-info">${data.query}</p>`)

                    data.context.forEach((element, index) => {
                        $('.result span').append(`Document ${index + 1}: <p class="text-success">${element}</p><br>`);
                    });
                }
                else{

                    $('.result span').html("No response available at the moment.");

                }
            },
            error: function(xhr, status, error) {
                console.error('AJAX Error:', error);
                console.log('Response Text:', xhr.responseText); // Ajout pour le débogage
            }
        });
        
            
        
        e.preventDefault();
        
    })

    // ----------------------------------------------------------------------------------------

    // Function to start listening to the event stream
    function startAgentEventStream() {
        if (typeof(EventSource) !== "undefined") {
            var source = new EventSource("/stream-agent");
            
            let last_node = ""
            let nodeText = ""

            source.onmessage = function(event) {
                // Log the event data here
                const data = JSON.parse(event.data);
                console.log(data); // This logs the incoming data from the server

                log = data.result.log
                color = data.result.color
                node = data.node
                
                if(node === last_node){
                    nodeText = ""
                }
                else{
                    last_node = node
                    nodeText = node
                }

                if(data.node === ""){
                    $('.result span').html(`${nodeText}<p class="text-success" style="color:${color}!important">${log}</p><br>`)
                }
                else{
                    $('.result span').append(`${nodeText}<p class="text-success" style="color:${color}!important">${log}</p><br>`)
                }

                if(data.node === "Final Answer :"){
                    $('#loader').hide();

                    $('.response').html(log);

                    $('form textarea[name="query"]').val(data.query);

                    $('form input[name="temperature"]').val(data.temperature);
                    
                    $('form input[name="base_weight"]').val(data.base_weight);
                    
                    $('form input[name="base_n"]').val(data.base_n);

                    $('form input[name="bm25_n"]').val(data.bm25_n);
                
                    $('form input[name="max_iter"]').val(data.max_iter);

                    $('form input[name="max_retries"]').val(data.max_retries);

                    $('form select[name="target"]').val(data.target);

                    $('form select[name="chat_model"]').val(data.chat_model);

                    $('form select[name="tr_model"]').val(data.tr_model);

                    $('form select[name="embedding_id"]').val(data.embedding_id);
                    
                    $('form select[name="metric"]').val(data.metric);
                    
                    $('form select[name="reranker"]').val(data.reranker);

                    $('form textarea[name="q_prompt"]').val(data.q_prompt);

                    $('form textarea[name="c_prompt"]').val(data.c_prompt);
                    
                    $('form textarea[name="e_prompt"]').val(data.e_prompt);

                    $('form textarea[name="s_prompt"]').val(data.s_prompt);
                    
                    $('form textarea[name="d_prompt"]').val(data.d_prompt);
                }
                else if(data.node === "Error Criterion :"){
                    $('#loader').hide();

                    $('.response').html(`<i class="text-danger">${log}</i>`);

                    $('form textarea[name="query"]').val(data.query);

                    $('form input[name="temperature"]').val(data.temperature);

                    $('form input[name="base_weight"]').val(data.base_weight);
                    
                    $('form input[name="base_n"]').val(data.base_n);

                    $('form input[name="bm25_n"]').val(data.bm25_n);
                
                    $('form input[name="max_iter"]').val(data.max_iter);

                    $('form input[name="max_retries"]').val(data.max_retries);

                    $('form select[name="target"]').val(data.target);

                    $('form select[name="chat_model"]').val(data.chat_model);

                    $('form select[name="tr_model"]').val(data.tr_model);

                    $('form select[name="embedding_id"]').val(data.embedding_id);
                    
                    $('form select[name="metric"]').val(data.metric);
                    
                    $('form select[name="reranker"]').val(data.reranker);

                    $('form textarea[name="q_prompt"]').val(data.q_prompt);

                    $('form textarea[name="c_prompt"]').val(data.c_prompt);
                    
                    $('form textarea[name="e_prompt"]').val(data.e_prompt);

                    $('form textarea[name="s_prompt"]').val(data.s_prompt);
                    
                    $('form textarea[name="d_prompt"]').val(data.d_prompt);
                }
                
                
            };

            source.onerror = function(event) {
                console.error("EventSource failed:", event);
                source.close(); // Close the connection on error
            };
        } else {
            document.getElementById("result").innerHTML = "Sorry, your browser does not support server-sent events...";
        }
    }

    $('#agent-form button[type="submit"]').on('click', function(e){
        
        // Show the loader
        $('#loader').show();

        // Initialize result
        $('.result span').html("In waiting for a response ...")

        data_obj = {
            query: $('form textarea[name="query"]').val(),
            temperature: $('form input[name="temperature"]').val(),
            base_weight: $('form input[name="base_weight"]').val(),
            base_n: $('form input[name="base_n"]').val(),
            bm25_n: $('form input[name="bm25_n"]').val(),
            max_iter: $('form input[name="max_iter"]').val(),
            max_retries: $('form input[name="max_retries"]').val(),
            chat_model: $('form select[name="target"]').val(),
            chat_model: $('form select[name="chat_model"]').val(),
            tr_model: $('form select[name="tr_model"]').val(),
            embedding_id: $('form select[name="embedding_id"]').val(),
            metric: $('form select[name="metric"]').val(),
            reranker: $('form select[name="reranker"]').val(),
            q_prompt: $('form textarea[name="q_prompt"]').val(),
            c_prompt: $('form textarea[name="c_prompt"]').val(),
            e_prompt: $('form textarea[name="e_prompt"]').val(),
            s_prompt: $('form textarea[name="s_prompt"]').val(),
            d_prompt: $('form textarea[name="d_prompt"]').val(),
        }

        $.ajax({
            data : data_obj,
            type : 'POST',
            url : '/agent_system/',
            success: function (data){
                console.log(data)

                $('title').text(data.title);

                if(data.correct === true){

                    $('.result span').html("In waiting for a response ...");

                    startAgentEventStream()

                }
                else{

                    // Hide the loader
                    $('#loader').hide();

                    $('.result span').html("No response available at the moment.");

                    $('.response').html(data.response);

                }
            },
            error: function(xhr, status, error) {
                console.error('AJAX Error:', error);
                console.log('Response Text:', xhr.responseText); // Ajout pour le débogage
            }
                
        });
            
        
        e.preventDefault();
        
    })
   
  });