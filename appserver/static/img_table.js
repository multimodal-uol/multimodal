require([
    'splunkjs/mvc',
	'splunkjs/mvc/searchmanager',
	'splunkjs/mvc/postprocessmanager',
    'splunkjs/mvc/tableview',
    'underscore',
    "splunkjs/mvc/simplexml/ready!"
], function(mvc,SearchManager,PostProcessManager,TableView,_) { 

	new SearchManager({
		id: "img_table_search",
		earliest_time: "0",
		latest_time: "now",
		preview: true,
		cache: false,
		search: "| inputlookup text_img.csv | table orig_text, twitter_img_urls, newspaper_img_urls" 
	});

    var myCustomtable = new TableView({
        id: "example-table",
        managerid: "img_table_search",
        pageSize: "10",
        el: $("#img_table")
    }).render();

    // Inherit from the BaseCellRenderer base class
    var MyCustomCellRenderer = TableView.BaseCellRenderer.extend({
        initialize: function() {
           //
        },
        canRender: function(cellData) {
            // Required
             return cellData.field === "orig_text" || cellData.field === "twitter_img_urls" || cellData.field === "newspaper_img_urls" 
        },
        setup: function($td, cellData) {
            //
        },
        teardown: function($td, cellData) {
            //
        },
        render: function($td, cellData) {
            // Required
            if (cellData.field === "orig_text") {
                $td.addClass('some_class1').html(_.template(
                    `<p align=left><%-orig_text%></p>`
                    ,{orig_text:cellData.value}
                ))
            }
            
            if (cellData.field === "twitter_img_urls" || cellData.field === "newspaper_img_urls") {
                image_url = cellData.value
                $td.addClass('some_class').html(_.template(
                    `<a href="<%-image_url%>">
                        <img src="<%-image_url%>" style="width: 30vw; min-width: 220px;"></img>
                     </a>`
                ,{image_url:image_url}))
            }
            if (cellData.field === "overview") {
                $td.addClass('some_class1').html(_.template(
                    `<p align=left><%-overview%></p>`
                    ,{overview:cellData.value}
                ))
            }
        }
    });

    var myCellRenderer = new MyCustomCellRenderer();

    myCustomtable.addCellRenderer(myCellRenderer); 

    myCustomtable.render();
    
    // TODO: When the $specific_word$ token changes, load the table again
//     var defaultTokenModel = mvc.Components.get("default");
//     defaultTokenModel.on("change:specific_word", function() {
//             console.log("specific_word: ",specific_word);
//     });
    


});