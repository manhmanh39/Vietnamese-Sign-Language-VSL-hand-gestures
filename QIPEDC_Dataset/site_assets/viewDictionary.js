$(document).ready(function() {
    load(20, '');
});


function onSearch() {
    var gr = $('#group').val();
    load(gr, $('#myInput').val().trim());
    // console.log($('#myInput').val().trim());
}

var input = document.getElementById("myInput");
input.addEventListener("keyup", function(event) {
    if (event.keyCode === 13) {
        onSearch();
    }
});

function inputSearch() {
    if ($('#myInput').val().trim() == "") {
        onSearch();
    }
}

function load(group, text) {
    $.ajax({
        type: "POST",
        url: "/dictionary/getAll",
        data: {
            group: group,
            text: text,
        },
        success: function(data) {
            // console.log(text);
            var data = data.data;
            // console.log(data);
            let perPage = group;
            let currentPage = 1;
            if (typeof(Storage) !== 'undefined') {
                // get sessionStorage
                var cp = sessionStorage.getItem('currentPage');
                if (cp !== 'undefined') {
                    currentPage = parseInt(cp + '');
                }
            }
            if (currentPage == null || currentPage == 0 || isNaN(currentPage)) currentPage = 1;
            // console.log(currentPage);
            let start = 0;
            let end = perPage;
            if (end > data.length) end = data.length;
            const totalPages = Math.ceil(data.length / perPage);
            if (currentPage > totalPages) currentPage = totalPages;
            if (data.length == 0) {
                document.getElementById('product').innerHTML = '<div class="col text-center">Không có kết quả</div>';
                document.getElementById('pagination-wrapper').innerHTML = '';
                return;
            }

            function getCurrentPage(currentPage) {
                start = (currentPage - 1) * perPage;
                end = currentPage * perPage;
                if (end > data.length) end = data.length;
            }
            // hiển thị lần lượt trong id product 
            function renderProduct() {
                var html = '';
                for (var i = start; i < end; i++) {
                    var item = data[i];
                    html += '<a class="col-xl-3 col-lg-3 col-md-4 t-a-center col-sm-6 hover-scale hover py-3" onclick="modalData(' + "'" + item._id + "'," + "'" + item.word + "'," + "'" + item.description + "'," + "'" + item.i + "'" + ' )" data-toggle="modal" data-target="#video-modal">'
                    html += ' <img src="/thumbs/' + item._id + '.png" class="m-w-100"></video>'
                    html += '<p class="t-a-center f-s-18 f-f-Lato-Black mb-0 ">' + item.word + '</p>'
                    html += '</a>'
                }
                document.getElementById('product').innerHTML = html;
            }
            getCurrentPage(currentPage);
            renderProduct();

            var state = {
                'querySet': data,
                'page': currentPage,
                'rows': perPage,
                'window': 6,
            }
            buildTable(currentPage);

            function pagination(querySet, page, rows) {
                var trimStart = (page - 1) * rows
                var trimEnd = trimStart + rows
                var trimmedData = querySet.slice(trimStart, trimEnd)
                var pages = Math.ceil(querySet.length / rows);
                return {
                    'querySet': trimmedData,
                    'pages': pages,
                }
            }

            function pageButtons(pages, currentPage) {
                var wrapper = document.getElementById('pagination-wrapper')
                // console.log(wrapper);
                wrapper.innerHTML = '';
                // console.log('Pages:', pages)

                var maxLeft = (state.page - Math.floor(state.window / 2))
                var maxRight = (state.page + Math.floor(state.window / 2))

                if (maxLeft < 1) {
                    maxLeft = 1
                    maxRight = state.window
                }
                if (maxRight > pages) {
                    maxLeft = pages - (state.window - 1)
                    if (maxLeft < 1) {
                        maxLeft = 1
                    }
                    maxRight = pages
                }
                for (var page = maxLeft; page <= maxRight; page++) {
                    wrapper.innerHTML +=
                        (currentPage == page) ?
                        ('<button value=' + page + '  class="page btn mx-1 btn-sm btn-info  ">' + page + '</button>') :
                        ('<button value=' + page + '  class="page btn mx-1 btn-sm btn-outline-info  ">' + page + '</button>')
                }
                if (state.page != 1) {
                    wrapper.innerHTML = '<button value=1 class="page btn mx-1 btn-sm btn-outline-info">&#171; First</button>' + wrapper.innerHTML
                }
                if (state.page != pages) {
                    wrapper.innerHTML += '<button value=' + pages + ' class="page btn mx-1 btn-sm btn-outline-info">Last &#187;</button>'
                }
                $('.page').on('click', function() {
                    currentPage = state.page = Number($(this).val())
                    // console.log(state.page);
                    // console.log(currentPage);
                    getCurrentPage(currentPage);
                    renderProduct();
                    buildTable(currentPage)
                    if (typeof(Storage) !== 'undefined') {
                        // Khởi tạo sesionStorage
                        sessionStorage.setItem('currentPage', currentPage);
                    }
                })
            }

            function buildTable(currentPage) {
                var data = pagination(state.querySet, state.page, state.rows)
                pageButtons(data.pages, currentPage)
            }
        }
    });
}


function modalData(_data, title, sub, i) {
    var name = document.getElementById('exampleModalLabel');
    var video = document.getElementById('s_expert');
    var explain = document.getElementById('explain');
    var img = document.getElementById('img-dictionary');

    if (i == "true") {
        img.style.display = "";
        img.src = '/Anh/' + _data.substring(0, 5) + '.png';
        // console.log(i);
    } else {
        img.style.display = "none";
    }
    name.innerHTML = title;
    explain.innerHTML = sub;
    video.src = '/videos/' + _data + '.mp4?autoplay=true';
    // console.log(1);
}

function closeModal() {
    var video = document.getElementById('s_expert');
    video.src = '';
    explain.innerHTML = '';
}