$(document).ready(function() {
    
    // ----------------------------------------------------------------------------------------
    
    // $('form.rag .left-choices select').on('change', function(e){
    //     data_obj = {
    //         domaine : $('form.mixtral-agent .left-choices select[name="domaine"]').val(),
    //         loi : $('form.mixtral-agent .left-choices select[name="loi"]').val(),
    //         decret : $('form.mixtral-agent .left-choices select[name="decret"]').val(),
    //         arrete : $('form.mixtral-agent .left-choices select[name="arrete"]').val(),
    //         declaration : $('form.mixtral-agent .left-choices select[name="declaration"]').val(),
    //         partie : $('form.mixtral-agent .left-choices select[name="partie"]').val(),
    //         livre : $('form.mixtral-agent .left-choices select[name="livre"]').val(),
    //         titre : $('form.mixtral-agent .left-choices select[name="titre"]').val(),
    //         sous_titre : $('form.mixtral-agent .left-choices select[name="sous_titre"]').val(),
    //         chapitre : $('form.mixtral-agent .left-choices select[name="chapitre"]').val(),
    //         section : $('form.mixtral-agent .left-choices select[name="section"]').val(),
    //         sous_section : $('form.mixtral-agent .left-choices select[name="sous_section"]').val(),
    //         application : $('form.mixtral-agent .left-choices select[name="application"]').val(),
    //         loyer : $('form.mixtral-agent .left-choices select[name="loyer"]').val(),
    //         localite : $('form.mixtral-agent .left-choices select[name="localite"]').val(),
    //         categorie : $('form.mixtral-agent .left-choices select[name="categorie"]').val(),
    //         habitation : $('form.mixtral-agent .left-choices select[name="habitation"]').val()
    //     }

    //     $.ajax({
    //         data : data_obj,
    //         type : 'POST',
    //         url : '/rag_system/'
    //     })
    //     .done(function (data){
    //         console.log(data)
    //         const keys = Object.keys(data)
            
    //         keys.forEach(key => {
                
    //             $(`form.rag .left-choices select[name="${key}"]`).empty()
                
    //             data[key].forEach((element, i) => {
                    
    //                 let newOption = "";
    //                 if (element == data_obj[key]){
    //                     newOption = $(`<option selected></option>`)
    //                             .attr('value', element)
    //                             .text(element);
    //                 }
    //                 else{
    //                     newOption = $('<option></option>')
    //                             .attr('value', element)
    //                             .text(element);
    //                 }
    
    //                 $(`form.rag .left-choices select[name="${key}"]`).append(newOption)
    //             });
                
    //         });
            
    //     });
    //     e.preventDefault();
        
    // })

    $('#rag-form button[type="submit"]').on('click', function(e){
        
        // Show the loader
        $('#loader').show();

        // Initialize result
        $('.result span').html("En Attente de Réponse ...")

        data_obj = {
            // domaine : $('form.mixtral-agent .left-choices select[name="domaine"]').val(),
            // loi : $('form.mixtral-agent .left-choices select[name="loi"]').val(),
            // decret : $('form.mixtral-agent .left-choices select[name="decret"]').val(),
            // arrete : $('form.mixtral-agent .left-choices select[name="arrete"]').val(),
            // declaration : $('form.mixtral-agent .left-choices select[name="declaration"]').val(),
            // partie : $('form.mixtral-agent .left-choices select[name="partie"]').val(),
            // livre : $('form.mixtral-agent .left-choices select[name="livre"]').val(),
            // titre : $('form.mixtral-agent .left-choices select[name="titre"]').val(),
            // sous_titre : $('form.mixtral-agent .left-choices select[name="sous_titre"]').val(),
            // chapitre : $('form.mixtral-agent .left-choices select[name="chapitre"]').val(),
            // section : $('form.mixtral-agent .left-choices select[name="section"]').val(),
            // sous_section : $('form.mixtral-agent .left-choices select[name="sous_section"]').val(),
            // application : $('form.mixtral-agent .left-choices select[name="application"]').val(),
            // loyer : $('form.mixtral-agent .left-choices select[name="loyer"]').val(),
            // localite : $('form.mixtral-agent .left-choices select[name="localite"]').val(),
            // categorie : $('form.mixtral-agent .left-choices select[name="categorie"]').val(),
            // habitation : $('form.mixtral-agent .left-choices select[name="habitation"]').val(),
            query: $('form textarea[name="query"]').val(),
            temperature: $('form input[name="temperature"]').val(),
            base_n: $('form input[name="base_n"]').val(),
            bm25_n: $('form input[name="bm25_n"]').val(),
            chat_model: $('form select[name="chat_model"]').val(),
            tr_model: $('form select[name="tr_model"]').val(),
            embedding_id: $('form select[name="embedding_id"]').val(),
            metric: $('form select[name="metric"]').val(),
            reranker: $('form select[name="reranker"]').val(),
            c_prompt: $('form textarea[name="c_prompt"]').val(),
            e_prompt: $('form textarea[name="e_prompt"]').val(),
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
                
                $('form input[name="base_n"]').val(data.base_n);

                $('form input[name="bm25_n"]').val(data.bm25_n);

                $('form select[name="chat_model"]').val(data.chat_model);
                
                $('form select[name="tr_model"]').val(data.tr_model);

                $('form select[name="embedding_id"]').val(data.embedding_id);
                    
                $('form select[name="metric"]').val(data.metric);
                
                $('form select[name="reranker"]').val(data.reranker);

                $('form textarea[name="c_prompt"]').val(data.c_prompt);

                $('form textarea[name="e_prompt"]').val(data.e_prompt);
                
                if(data.correct){

                    $('.result span').html("")

                    data.context.forEach((element, index) => {
                        $('.result span').append(`Document ${index + 1}: <p class="text-success">${element}</p><br>`);
                    });
                }
                else{

                    $('.result span').html("Pas de Réponse Disponible pour le Moment.");

                }
        
                // $('#vector-response').html(`
                //     <div class="card-body">
                //         <div class='form-group blue-form'>
                //             <label class="card-title">Réponse :</label>
                //             <p class="card-text p-4 border">${data.response}</p>
                //         </div>
                //         <hr>
                //         <div class='form-group blue-form'>
                //             <label class="card-title">Requête :</label>
                //             <p class="card-text p-4 border">${data.query}</p>
                //         </div>
                //     </div>
                // `);
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

                if(data.node === "Filtration de Documents :"){
                    $('.result span').html(`${nodeText}<p class="text-success" style="color:${color}!important">${log}</p><br>`)
                }
                else{
                    $('.result span').append(`${nodeText}<p class="text-success" style="color:${color}!important">${log}</p><br>`)
                }

                if(data.node === "Réponse Finale :"){
                    $('#loader').hide();

                    $('.response').html(log);

                    $('form textarea[name="query"]').val(data.query);

                    $('form input[name="temperature"]').val(data.temperature);
                    
                    $('form input[name="base_n"]').val(data.base_n);

                    $('form input[name="bm25_n"]').val(data.bm25_n);
                
                    $('form input[name="max_iter"]').val(data.max_iter);

                    $('form select[name="chat_model"]').val(data.chat_model);

                    $('form select[name="tr_model"]').val(data.tr_model);

                    $('form select[name="embedding_id"]').val(data.embedding_id);
                    
                    $('form select[name="metric"]').val(data.metric);
                    
                    $('form select[name="reranker"]').val(data.reranker);

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

    // $('form.agent .left-choices select').on('change', function(e){
    //     data_obj = {
    //         domaine : $('form.mixtral-agent .left-choices select[name="domaine"]').val(),
    //         loi : $('form.mixtral-agent .left-choices select[name="loi"]').val(),
    //         decret : $('form.mixtral-agent .left-choices select[name="decret"]').val(),
    //         arrete : $('form.mixtral-agent .left-choices select[name="arrete"]').val(),
    //         declaration : $('form.mixtral-agent .left-choices select[name="declaration"]').val(),
    //         partie : $('form.mixtral-agent .left-choices select[name="partie"]').val(),
    //         livre : $('form.mixtral-agent .left-choices select[name="livre"]').val(),
    //         titre : $('form.mixtral-agent .left-choices select[name="titre"]').val(),
    //         sous_titre : $('form.mixtral-agent .left-choices select[name="sous_titre"]').val(),
    //         chapitre : $('form.mixtral-agent .left-choices select[name="chapitre"]').val(),
    //         section : $('form.mixtral-agent .left-choices select[name="section"]').val(),
    //         sous_section : $('form.mixtral-agent .left-choices select[name="sous_section"]').val(),
    //         application : $('form.mixtral-agent .left-choices select[name="application"]').val(),
    //         loyer : $('form.mixtral-agent .left-choices select[name="loyer"]').val(),
    //         localite : $('form.mixtral-agent .left-choices select[name="localite"]').val(),
    //         categorie : $('form.mixtral-agent .left-choices select[name="categorie"]').val(),
    //         habitation : $('form.mixtral-agent .left-choices select[name="habitation"]').val()
    //     }

    //     $.ajax({
    //         data : data_obj,
    //         type : 'POST',
    //         url : '/agent_system/'
    //     })
    //     .done(function (data){
    //         console.log(data)
    //         const keys = Object.keys(data)
            
    //         keys.forEach(key => {
                
    //             $(`form.agent .left-choices select[name="${key}"]`).empty()
                
    //             data[key].forEach((element, i) => {
                    
    //                 let newOption = "";
    //                 if (element == data_obj[key]){
    //                     newOption = $(`<option selected></option>`)
    //                             .attr('value', element)
    //                             .text(element);
    //                 }
    //                 else{
    //                     newOption = $('<option></option>')
    //                             .attr('value', element)
    //                             .text(element);
    //                 }
    
    //                 $(`form.agent .left-choices select[name="${key}"]`).append(newOption)
    //             });
                
    //         });
            
    //     });
    //     e.preventDefault();
        
    // })

    $('#agent-form button[type="submit"]').on('click', function(e){
        
        // Show the loader
        $('#loader').show();

        // Initialize result
        $('.result span').html("En Attente de Réponse ...")

        data_obj = {
            // domaine : $('form.mixtral-agent .left-choices select[name="domaine"]').val(),
            // loi : $('form.mixtral-agent .left-choices select[name="loi"]').val(),
            // decret : $('form.mixtral-agent .left-choices select[name="decret"]').val(),
            // arrete : $('form.mixtral-agent .left-choices select[name="arrete"]').val(),
            // declaration : $('form.mixtral-agent .left-choices select[name="declaration"]').val(),
            // partie : $('form.mixtral-agent .left-choices select[name="partie"]').val(),
            // livre : $('form.mixtral-agent .left-choices select[name="livre"]').val(),
            // titre : $('form.mixtral-agent .left-choices select[name="titre"]').val(),
            // sous_titre : $('form.mixtral-agent .left-choices select[name="sous_titre"]').val(),
            // chapitre : $('form.mixtral-agent .left-choices select[name="chapitre"]').val(),
            // section : $('form.mixtral-agent .left-choices select[name="section"]').val(),
            // sous_section : $('form.mixtral-agent .left-choices select[name="sous_section"]').val(),
            // application : $('form.mixtral-agent .left-choices select[name="application"]').val(),
            // loyer : $('form.mixtral-agent .left-choices select[name="loyer"]').val(),
            // localite : $('form.mixtral-agent .left-choices select[name="localite"]').val(),
            // categorie : $('form.mixtral-agent .left-choices select[name="categorie"]').val(),
            // habitation : $('form.mixtral-agent .left-choices select[name="habitation"]').val(),
            query: $('form textarea[name="query"]').val(),
            temperature: $('form input[name="temperature"]').val(),
            base_n: $('form input[name="base_n"]').val(),
            bm25_n: $('form input[name="bm25_n"]').val(),
            max_iter: $('form input[name="max_iter"]').val(),
            chat_model: $('form select[name="chat_model"]').val(),
            tr_model: $('form select[name="tr_model"]').val(),
            embedding_id: $('form select[name="embedding_id"]').val(),
            metric: $('form select[name="metric"]').val(),
            reranker: $('form select[name="reranker"]').val(),
            c_prompt: $('form textarea[name="c_prompt"]').val(),
            e_prompt: $('form textarea[name="e_prompt"]').val(),
            s_prompt: $('form textarea[name="s_prompt"]').val(),
            d_prompt: $('form textarea[name="d_prompt"]').val(),
        }

        $.ajax({
            data : data_obj,
            type : 'POST',
            url : '/agent_system/',
            complete: function() {
                // Hide the loader when the request is complete
                startAgentEventStream()
            },
            success: function (data){
                console.log(data)

                $('title').text(data.title);

                if(data.correct === true){

                    $('.result span').html("En Attente de Réponse ...");

                }
                else{

                    $('.result span').html("Pas de Réponse Disponible pour le Moment.");

                }
                // $('title').text(data.title)

            //     $('#vector-response').html(`<div class="card-body">
                
            //     <div class='form-group blue-form'>
            //         <label class="card-title">Réponse :</label>
            //         <p class="card-text p-4 border">${data.response}</p>
            //     </div>
            //     <hr>
            //     <div class='form-group blue-form'>
            //         <label class="card-title">Requête :</label>
            //         <p class="card-text p-4 border">${data.query}</p>
            //     </div>
            // </div>`)
            },
            error: function(xhr, status, error) {
                console.error('AJAX Error:', error);
                console.log('Response Text:', xhr.responseText); // Ajout pour le débogage
            }
                
        });
            
        
        e.preventDefault();
        
    })
   
  });