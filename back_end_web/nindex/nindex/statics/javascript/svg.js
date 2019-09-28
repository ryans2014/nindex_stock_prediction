var margin = {top: 30, right: 30, bottom: 40, left: 50};
var width = d3.select(".navbar").node().clientWidth * 0.8;
if (screen.width * 1.2 < screen.height) {
	width = d3.select(".navbar").node().clientWidth;
}
var height = width / 960.0 * 600.0
width = width - margin.left - margin.right,
height = height - margin.top - margin.bottom;

var svg = d3.select("body").select("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)

var svg_g = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// DOM axis objects
var gX = svg_g.append("g")
    .attr("class", "axis axis--x")
    .attr("transform", "translate(0," + height + ")");

var gY = svg_g.append("g")
    .attr("class", "axis axis--y");

gY.append("text")
    .attr("class", "axis-title")
    .attr("transform", "rotate(-90)")
    .attr("y", 6)
    .attr("dy", ".71em")

// d3 objects
// var parseDate = d3.timeParse("%y-%b-%d");
var parseDate = d3.timeParse("%Y-%m-%d");
var x = d3.scaleTime().range([0, width]);
var y = d3.scaleLinear().range([height, 0]);

var xAxis = d3.axisBottom(x);
var yAxis = d3.axisLeft(y)

var zoomX = d3.zoom()
    .scaleExtent([1, 32])
    .translateExtent([[0, 0], [width, height]])
    .extent([[0, 0], [width, height]])
    .on("zoom", zoomed_xaxis);

var line = d3.line()
    .x(function(d) { return x(d.date); })
    .y(function(d) { return y(d.y1); });

var area_above = d3.area()
    .x(function(d) { return x(d.date); })
    .y0(function(d) { return y(d.y1); })
    .y1(function(d) { return y(d.y2); });

var area_below = d3.area()
    .x(function(d) { return x(d.date); })
    .y0(function(d) { return y(d.y0); })
    .y1(function(d) { return y(d.y1); });

var x_range = [];
var data = [];


// get csv string and process
var xhttp = new XMLHttpRequest();
xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
        var csv_string = xhttp.responseText;
        // read data and make plots
        var rows = d3.csvParse(csv_string, function(d) { return {date: d.date, close: +d.close, predict: +d.predict}; });
        data = rows;

        // remove loading_gif
        d3.select(".loading_gif").attr("src","")

        // check error
        if (rows.length < 10) {
            d3.select("#svg_place_holder").text("Error loading stock prediction. Please check if the entered stock symbol is valid.")
            d3.select("#percentage_result").text("Unavailable ..");
            return;
        }

        // insert prediction
        var cg = data[data.length - 1].predict  / data[data.length - 1].close * 100;
        cg = String(cg).slice(0,4) + "%"
        if (cg[0] != "-") {
            cg = "+" + cg;
        }
        d3.select("#percentage_result").text(cg);

	// process data
        rows.forEach(function(row) {
            row.date = parseDate(row.date);
        });
        rows.forEach(function(row) {
            row.y1 = row.close;
            row.y0 = d3.min([row.close, row.close + row.predict]);
            row.y2 = d3.max([row.close, row.close + row.predict]);
        });

        // set input domain
        x.domain(d3.extent(rows, function(row) { return row.date; }));
        y.domain(d3.extent(rows, function(row) { return row.y1; }));

        // set axis
        yAxis.tickValues(d3.scaleLinear().domain(y.domain()).ticks(20));
        gX.call(xAxis);
        gY.call(yAxis)
            .selectAll(".tick")
            .classed("tick--one", function(d) { return Math.abs(d - 1) < 1e-6; });

        // generate line plot and area plot
        var defs = svg_g.append("defs");

        svg_g.append("path")
            .datum(rows)
            .attr("class", "area area--above")
            .attr("d", area_above);

        svg_g.append("path")
            .datum(rows)
            .attr("class", "area area--below")
            .attr("d", area_below);

        svg_g.append("path")
            .datum(rows)
            .attr("class", "line")
            .attr("d", line);

        // register zooming
        // d3.select("svg").call(zoom);
        var d0 = rows[rows.length - d3.min([356, rows.length])].date,
            d1 = rows[rows.length - 1].date;

        // Gratuitous intro zoom
        svg.call(zoomX).transition()
            .duration(1500)
            .call(zoomX.transform, d3.zoomIdentity
            .scale(width / (x(d1) - x(d0)))
            .translate(-x(d0), 0));

    }
};

var url = window.location.href;
url = url.replace("result", "result/csv");
if (url.slice(-1) == "/") {
	url = url.slice(0, url.length - 1);
}
xhttp.open("GET", url, true);
xhttp.send();

function zoomed_xaxis() {
    var t = d3.event.transform, xt = t.rescaleX(x);
    x_range = xt.domain();
    d3.select(".area--above").attr("d", area_above.x(function(d) { return xt(d.date); }))
    d3.select(".area--below").attr("d", area_below.x(function(d) { return xt(d.date); }))
    d3.select(".line").attr("d", line.x(function(d) { return xt(d.date); }))
    gX.call(xAxis.scale(xt));
}

function fit_yaxis() {
    var range_data = data.filter(function(d) { return d.date >= x_range[0] && d.date <= x_range[1];});
    var domain = extend_range(d3.extent(range_data, function(d) { return d.y1; }));
    var yt = y.copy().domain(domain);
    d3.select(".area--above").attr("d", area_above.y0(function(d) { return yt(d.y1); }))
                           .attr("d", area_above.y1(function(d) { return yt(d.y2); }));
    d3.select(".area--below").attr("d", area_below.y0(function(d) { return yt(d.y0); }))
                           .attr("d", area_below.y1(function(d) { return yt(d.y1); }));
    d3.select(".line").attr("d", line.y(function(d) { return yt(d.y1); }));
    gY.call(yAxis.scale(yt));
}

function extend_range(range) {
    var delta = range[1] - range[0];
    range[0] = range[0] - delta / 6;
    range[1] = range[1] + delta / 6;
    return range;
}

function fit_all() {
    d3.select(".area--above").attr("d", area_above.y0(function(d) { return y(d.y1); }))
                         .attr("d", area_above.y1(function(d) { return y(d.y2); }));
    d3.select(".area--below").attr("d", area_below.y0(function(d) { return y(d.y0); }))
                         .attr("d", area_below.y1(function(d) { return y(d.y1); }));
    d3.select(".line").attr("d", line.y(function(d) { return y(d.y1); }));
    gY.call(yAxis.scale(y));
    d3.select(".area--above").attr("d", area_above.x(function(d) { return x(d.date); }))
    d3.select(".area--below").attr("d", area_below.x(function(d) { return x(d.date); }))
    d3.select(".line").attr("d", line.x(function(d) { return x(d.date); }))
    gX.call(xAxis.scale(x));
    x_range = x.domain();
}

d3.select("#fit_y_axis").on("click", fit_yaxis);
d3.select("#fit_all").on("click", fit_all);
