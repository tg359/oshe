# Outdoor Sun/Sky/Surface Heat Exchange (OSHE)

The repo contains a full example and code for simulating outdoor heat exchange between sample points, contextual geometry surfaces (with material properties), and the sun and sky. The underlying code for this relies on the [Ladybug SolarCal](https://github.com/ladybug-tools/ladybug-comfort/blob/master/ladybug_comfort/solarcal.py) method calculating an effective radiant field (ERF) to convert into a radiant temperature uplift, with some slight modifications to enable Radiance radiation results to be passed into it. The resultant overall radiant temperature can then be plugged into the Universal Thermal Climate Index (other comfort indices are available) to assess comfort levels throughout the year at the sample point locations.
 
The full process is as follows:
1. Define geometry and materials in the Rhino model (see `./example/RH6_model.3dm` for an example structure). The important things to remember for this are that the ground zone should be a closed Brep, and all geometry needs to be clean or coincident vertices, nurbs-curve boundaries or degenerate faces.
2. Run the `./GenerateCase.gh` script to create the `runenergyplus.bat` and `runradiance.bat` files, then while those are running create the `recipe.json` file containing point-surface-sky view factors. It is worth using the simplified EnergyPlus simulation first to check that a valid simulatable model has been created!
3. Once the simulations are complete, run the `PostProcess.ipynb` script to load the `recipe.json`, annual point-radiation results from Radiance and annual surface temperature results from EnergyPlus.
4. Open field MRT is calculated from a simple case of unshaded exposed ground of given type (default is "CONCRETE").
5. The MRT for each point is calculated using the view-factors and surface temperature to surfaces, view factor to sky and sky temperature, and radiation from sun and reflected from context in the Radiance results. This method is split across n-processes to help speed up the processing of large numbers of sample points.
6. The UTCI for each point is also calculated, again across multiple processes to speed up processing.
7. A UTCI object is created containing the open-field results and the point-results, which can then be used to plot the time-filtered performance of the sample area. 

The outputs returned from this whole process, are annual-hourly point-wise MRT and UTCI for the sample-points, and a set of plots detailing the comfort of the area assessed, and focus-point hourly performance.

<div class="tg-wrap">
    <table>
        <tbody>
            <tr>
                <td>
                    <img src="https://github.com/tg359/oshe/blob/ghadan21/example/plots/reduction_may_morningshoulder.png" alt="May morning UTCI comfort improvement"> 
                </td>
                <td>
                    <img src="https://github.com/tg359/oshe/blob/ghadan21/example/plots/comfortable_hours_annual.png" alt="Annual comfortable hours - comparison between open field and sampled points">
                </td>
                <td>
                    <img src="https://github.com/tg359/oshe/blob/ghadan21/example/plots/context_focuspts.png" alt="Context geometry and focus points"> 
                </td>
            </tr>
            <tr>
                <td colspan="3">
                    <img src="https://github.com/tg359/oshe/blob/ghadan21/example/plots/pt0001_collected.png" alt="Focus point open field comparison"> 
                </td>
            </tr>
        </tbody>
    </table>
</div>